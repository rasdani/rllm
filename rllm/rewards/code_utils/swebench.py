from __future__ import annotations

import json
import platform
import traceback

import docker

if platform.system() == "Linux":
    import resource

import uuid
from pathlib import Path, PurePosixPath

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    UTF8,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.docker_utils import (
    clean_images,
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
    remove_image,
    should_remove,
)
from swebench.harness.grading import get_eval_report

# from swebench.harness.reporting import make_run_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import (
    EvaluationError,
    load_swebench_dataset,
    run_threadpool,
)

from rllm.globals import CACHE_LEVEL, CLEAN, FORCE_REBUILD, INSTANCE_IMAGE_TAG, MAX_WORKERS, MODEL_NAME_OR_PATH, NAMESPACE, OPEN_FILE_LIMIT, REPORT_DIR, REWRITE_REPORTS, SPLIT, SWEBENCH_DATASET_NAME, TIMEOUT

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def run_instance(
    test_spec: TestSpec,
    pred: dict,
    rm_image: bool,
    force_rebuild: bool,
    client: docker.DockerClient,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False,
):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
        rewrite_reports (bool): True if eval run is just to reformat existing report
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

    # Set up report file
    report_path = log_dir / LOG_REPORT
    if rewrite_reports:
        test_output_path = log_dir / LOG_TEST_OUTPUT
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())

    if not test_spec.is_remote_image:
        # Link the image build dir in the log dir
        build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
        image_build_link = log_dir / "image_build_dir"
        if not image_build_link.exists():
            try:
                # link the image build dir in the log dir
                image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
            except:
                # some error, idk why
                pass

    # Set up logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    try:
        # Build + start instance container (instance image should already be built)
        container = build_container(test_spec, client, run_id, logger, rm_image, force_rebuild)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        logger.info(f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container...")
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        # Attempt to apply patch to container (TODO: FIX THIS)
        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = container.exec_run(f"{git_apply_cmd} {DOCKER_PATCH}", workdir=DOCKER_WORKDIR, user=DOCKER_USER)
            if val.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8)}")
                applied_patch = True
                break
            else:
                logger.info(f"Failed to apply patch to container: {git_apply_cmd}")
        if not applied_patch:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8)}",
                logger,
            )

        # Get git diff before running eval script
        git_diff_output_before = container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(f"Eval script for {instance_id} written to {eval_file}; copying to container...")
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

        # Run eval script, write output to logs
        test_output, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
        test_output_path = log_dir / LOG_TEST_OUTPUT
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")
        with open(test_output_path, "w") as f:
            f.write(test_output)
            logger.info(f"Test output for {instance_id} written to {test_output_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        # Get git diff after running eval script (ignore permission changes)
        git_diff_output_after = container.exec_run("git -c core.fileMode=false diff", workdir=DOCKER_WORKDIR).output.decode(UTF8).strip()

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info("Git diff changed after running eval script")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )

        logger.info(f"report: {report}\nResult for {instance_id}: resolved: {report[instance_id]['resolved']}")

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = f"Error in evaluating model for {instance_id}: {e}\n{traceback.format_exc()}\nCheck ({logger.log_file}) for more information."
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


def run_instances(
    predictions: dict,
    instances: list,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    max_workers: int,
    run_id: str,
    timeout: int,
    namespace: str = None,
    instance_image_tag: str = "latest",
    rewrite_reports: bool = False,
):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    client = docker.from_env()
    test_specs = list(map(lambda instance: make_test_spec(instance, namespace=namespace, instance_image_tag=instance_image_tag), instances))

    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {tag for i in client.images.list(all=True) for tag in i.tags if tag in instance_image_ids}
    if not force_rebuild and len(existing_images):
        print(f"Found {len(existing_images)} existing instance images. Will reuse them.")

    # run instances in parallel
    payloads = []
    for test_spec in test_specs:
        payloads.append(
            (
                test_spec,
                predictions[test_spec.instance_id],
                should_remove(
                    test_spec.instance_image_key,
                    cache_level,
                    clean,
                    existing_images,
                ),
                force_rebuild,
                client,
                run_id,
                timeout,
                rewrite_reports,
            )
        )

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    run_threadpool(run_instance, payloads, max_workers)
    print("All instances run.")


def get_dataset_from_preds(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions: dict,
    run_id: str,
    rewrite_reports: bool,
    exclude_completed: bool = True,
):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    """
    # load dataset
    dataset = load_swebench_dataset(dataset_name, split)
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        # check that all instance IDs have predictions
        missing_preds = set(instance_ids) - set(predictions.keys())
        if missing_preds:
            print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs.")

    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(f"Some prediction IDs not found in dataset!\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}")
    if instance_ids:
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

    if rewrite_reports:
        # we only return instances that have existing test outputs
        test_output_ids = set()
        for instance in dataset:
            if instance[KEY_INSTANCE_ID] not in predictions:
                continue
            prediction = predictions[instance[KEY_INSTANCE_ID]]
            test_output_file = RUN_EVALUATION_LOG_DIR / run_id / prediction["model_name_or_path"].replace("/", "__") / prediction[KEY_INSTANCE_ID] / "test_output.txt"
            if test_output_file.exists():
                test_output_ids.add(instance[KEY_INSTANCE_ID])
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in prediction_ids and i[KEY_INSTANCE_ID] in test_output_ids]
        return dataset

    # check which instance IDs have already been run
    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            # skip instances without predictions
            continue

        prediction = predictions[instance[KEY_INSTANCE_ID]]

        report_file = RUN_EVALUATION_LOG_DIR / run_id / prediction[KEY_MODEL].replace("/", "__") / prediction[KEY_INSTANCE_ID] / LOG_REPORT
        if report_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]

    empty_patch_ids = {k for k, v in predictions.items() if v[KEY_PREDICTION] == "" or v[KEY_PREDICTION] is None}

    # filter dataset to only instances with predictions
    dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in prediction_ids and i[KEY_INSTANCE_ID] not in empty_patch_ids]
    return dataset


def make_run_report(
    predictions: dict,
    full_dataset: list,
    run_id: str,
    client: Optional[docker.DockerClient] = None,
) -> Path:
    """
    Make a final evaluation and run report of the instances that have been run.
    Also reports on images and containers that may still running if client is provided.
    Referenced from swe bench harness but includes additional information in report including individual test case pass count.
    Args:
        predictions (dict): Predictions dict generated by the model
        full_dataset (list): List of all instances
        run_id (str): Run ID
        client (docker.DockerClient): Docker client (optional)

    Returns:
        Path to report file
    """
    # instantiate sets to store IDs of different outcomes
    completed_ids = set()
    resolved_ids = set()
    error_ids = set()
    unstopped_containers = set()
    unremoved_images = set()
    unresolved_ids = set()
    incomplete_ids = set()
    # get instances with empty patches
    empty_patch_ids = set()

    # for reward calculation
    f2p_success = set()
    f2p_failure = set()
    p2p_success = set()
    p2p_failure = set()
    f2f_success = set()
    f2f_failure = set()
    p2f_success = set()
    p2f_failure = set()

    # iterate through dataset and check if the instance has been run
    for instance in full_dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in predictions:
            # skip instances without predictions
            incomplete_ids.add(instance_id)
            continue
        prediction = predictions[instance_id]
        if prediction.get(KEY_PREDICTION, None) in ["", None]:
            empty_patch_ids.add(instance_id)
            continue
        report_file = RUN_EVALUATION_LOG_DIR / run_id / prediction[KEY_MODEL].replace("/", "__") / prediction[KEY_INSTANCE_ID] / LOG_REPORT
        if report_file.exists():
            # If report file exists, then the instance has been run
            completed_ids.add(instance_id)
            report = json.loads(report_file.read_text())
            if report[instance_id]["resolved"]:
                # Record if the instance was resolved
                resolved_ids.add(instance_id)
            else:
                unresolved_ids.add(instance_id)

            tests_status_map = report[instance_id]["tests_status"]
            if tests_status_map:
                f2p_success.update(tests_status_map["FAIL_TO_PASS"]["success"])
                f2p_failure.update(tests_status_map["FAIL_TO_PASS"]["failure"])
                p2p_success.update(tests_status_map["PASS_TO_PASS"]["success"])
                p2p_failure.update(tests_status_map["PASS_TO_PASS"]["failure"])
                f2f_success.update(tests_status_map["FAIL_TO_FAIL"]["success"])
                f2f_failure.update(tests_status_map["FAIL_TO_FAIL"]["failure"])
                p2f_success.update(tests_status_map["PASS_TO_FAIL"]["success"])
                p2f_failure.update(tests_status_map["PASS_TO_FAIL"]["failure"])
        else:
            # Otherwise, the instance was not run successfully
            error_ids.add(instance_id)

    if client:
        # get remaining images and containers
        images = list_images(client)
        test_specs = list(map(make_test_spec, full_dataset))
        for spec in test_specs:
            image_name = spec.instance_image_key
            if image_name in images:
                unremoved_images.add(image_name)
        containers = client.containers.list(all=True)
        for container in containers:
            if run_id in container.name:
                unstopped_containers.add(container.name)

    # print final report
    dataset_ids = {i[KEY_INSTANCE_ID] for i in full_dataset}
    print(f"Total instances: {len(full_dataset)}")
    print(f"Instances submitted: {len(set(predictions.keys()) & dataset_ids)}")
    print(f"Instances completed: {len(completed_ids)}")
    print(f"Instances incomplete: {len(incomplete_ids)}")
    print(f"Instances resolved: {len(resolved_ids)}")
    print(f"Instances unresolved: {len(unresolved_ids)}")
    print(f"Instances with empty patches: {len(empty_patch_ids)}")
    print(f"Instances with errors: {len(error_ids)}")
    if client:
        print(f"Unstopped containers: {len(unstopped_containers)}")
        print(f"Unremoved images: {len(unremoved_images)}")

    # write report to file
    report = {
        "total_instances": len(full_dataset),
        "submitted_instances": len(predictions),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(predictions.keys())),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "f2p_success": len(f2p_success),
        "f2p_failure": len(f2p_failure),
        "p2p_success": len(p2p_success),
        "p2p_failure": len(p2p_failure),
        "f2f_success": len(f2f_success),
        "f2f_failure": len(f2f_failure),
        "p2f_success": len(p2f_success),
        "p2f_failure": len(p2f_failure),
        "schema_version": 2,
    }
    if not client:
        report.update(
            {
                "unstopped_instances": len(unstopped_containers),
                "unstopped_containers": list(sorted(unstopped_containers)),
                "unremoved_images": list(sorted(unremoved_images)),
            }
        )
    report_file = Path(list(predictions.values())[0][KEY_MODEL].replace("/", "__") + f".{run_id}" + ".json")
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")
    return report_file


def run_evaluation(
    dataset_name: str,
    instance_ids: list,
    actions: dict[str:str],
    max_workers: int,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    open_file_limit: int,
    run_id: str,
    timeout: int,
    namespace: str | None,
    rewrite_reports: bool,
    split: str = "test",
    instance_image_tag: str = "latest",
    report_dir: str = ".",
) -> Path:
    """
    Run evaluation harness for the given dataset and predictions.

    Args:
    - dataset_name (str): The name of the dataset to evaluate. (ex: princeton-nlp/SWE-bench_Lite)
    - instance_ids (list): List of instance IDs to evaluate.
    - actions (dict[str: str]): The action to perform during evaluation. It is a dictionary mapping instance IDs to actions.
    - max_workers (int): Maximum number of worker threads to use.
    - force_rebuild (bool): Whether to force rebuild the environment images.
    - cache_level (str): The level of caching to use.
    - clean (bool): Whether to clean up after evaluation.
    - open_file_limit (int): The limit for the number of open files.
    - run_id (str): Unique identifier for the run.
    - timeout (int): Timeout for the evaluation process.
    - namespace (str | None): Namespace to use for the evaluation.
    - rewrite_reports (bool): Whether to rewrite existing reports.
    - split (str, optional): The dataset split to use. Defaults to 'test'.
    - instance_image_tag (str, optional): Tag for the instance image. Defaults to 'latest'.
    - report_dir (str, optional): Directory to save reports. Defaults to '.'.

    Returns:
    dict: A report of the evaluation run.
    """

    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    if report_dir is not None:
        report_dir = Path(report_dir)
        if not report_dir.exists():
            report_dir.mkdir(parents=True)

    if force_rebuild and namespace is not None:
        raise ValueError("Cannot force rebuild and use a namespace at the same time.")

    # load predictions as map of instance_id to prediction

    # predictions = get_predictions_from_file(predictions_path, dataset_name, split)
    # predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}
    # predictions = actions

    # get dataset from predictions
    dataset = get_dataset_from_preds(dataset_name, split, instance_ids, actions, run_id, rewrite_reports)
    full_dataset = load_swebench_dataset(dataset_name, split, instance_ids)

    # run instances locally
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    existing_images = list_images(client)
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    else:
        # build environment images + run instances
        if namespace is None and not rewrite_reports:
            build_env_images(client, dataset, force_rebuild, max_workers)
        run_instances(
            actions,
            dataset,
            cache_level,
            clean,
            force_rebuild,
            max_workers,
            run_id,
            timeout,
            namespace=namespace,
            instance_image_tag=instance_image_tag,
            rewrite_reports=rewrite_reports,
        )

    # clean images + make final report
    clean_images(client, existing_images, cache_level, clean)

    # read from the report - generate the reward - return the reward
    return make_run_report(actions, full_dataset, run_id, client)


def swebench_check_correctness(
    model_response: str,
    metadata: dict,
) -> float:
    tests = metadata.get("tests", None)
    instance_id = tests.get("instance_id", None)
    # Attempt to parse a patch from the model response
    patch_start = model_response.find("diff --git")

    patch = ""

    if patch_start != -1:
        patch_end = model_response.find("```", patch_start)
        if patch_end == -1:
            patch_end = len(model_response)
        patch = model_response[patch_start:patch_end].strip()

    params = {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": MODEL_NAME_OR_PATH,
    }
    actions = {instance_id: params}

    # generate unique run id
    run_id = uuid.uuid4().hex
    instance_ids = [instance_id]

    eval_report_path = run_evaluation(SWEBENCH_DATASET_NAME, instance_ids, actions, MAX_WORKERS, FORCE_REBUILD, CACHE_LEVEL, CLEAN, OPEN_FILE_LIMIT, run_id, TIMEOUT, NAMESPACE, REWRITE_REPORTS, SPLIT, INSTANCE_IMAGE_TAG, REPORT_DIR)

    # read from eval report and get the correct/incorrect stats for reward calculation
    with open(eval_report_path) as f:
        eval_report = json.load(f)

        # Calculate reward based on correct/incorrect stats
        f2p_success = eval_report["f2p_success"]
        f2p_failure = eval_report["f2p_failure"]
        p2p_success = eval_report["p2p_success"]
        p2p_failure = eval_report["p2p_failure"]
        f2f_success = eval_report["f2f_success"]
        f2f_failure = eval_report["f2f_failure"]
        p2f_success = eval_report["p2f_success"]
        p2f_failure = eval_report["p2f_failure"]

        tests_passed = f2p_success + p2p_success + f2f_success + p2f_success
        total_tests = f2p_success + f2p_failure + p2p_success + p2p_failure + f2f_success + f2f_failure + p2f_success + p2f_failure

        print(f"Tests Passed: {tests_passed}")
        print(f"Total Tests: {total_tests}")

        resolve_rate = tests_passed / total_tests if total_tests > 0 else 0

        return resolve_rate
