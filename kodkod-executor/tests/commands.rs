#[cfg(unix)]
mod unix {
    use kodkod_executor::command::{CommandRequest, run};
    use std::{path::Path, time::Duration};
    use tempfile::tempdir;
    use tokio::time::{sleep, timeout};

    #[tokio::test]
    async fn returns_stdout_stderr_and_exit_code() {
        let directory = tempdir().unwrap();
        let output = run(CommandRequest::shell(
            "printf 'standard output'; printf 'standard error' >&2; exit 7",
            directory.path(),
        ))
        .await
        .unwrap();

        assert_eq!(output.exit_code, Some(7));
        assert_eq!(output.stdout, "standard output");
        assert_eq!(output.stderr, "standard error");
        assert_eq!(output.output_limit_bytes, 64 * 1024);
        assert!(!output.stdout_truncated);
        assert!(!output.stderr_truncated);
    }

    #[tokio::test]
    async fn drains_large_stdout_and_stderr_and_reports_truncation() {
        let directory = tempdir().unwrap();
        let script = "i=0; while [ \"$i\" -lt 128 ]; do printf '%4096s' ''; printf '%4096s' '' >&2; i=$((i + 1)); done";
        let request = CommandRequest::shell(script, directory.path())
            .with_timeout_ms(10_000)
            .with_output_limit(1024);
        let output = timeout(Duration::from_secs(5), run(request))
            .await
            .expect("the shell should finish while both output pipes are drained")
            .unwrap();

        assert_eq!(output.exit_code, Some(0));
        assert_eq!(output.stdout.len(), 1024);
        assert_eq!(output.stderr.len(), 1024);
        assert!(output.stdout_truncated);
        assert!(output.stderr_truncated);
    }

    #[tokio::test]
    async fn timeout_kills_the_child_process_group() {
        let directory = tempdir().unwrap();
        let script = "(printf started > timeout.started; sleep 1.5; printf completed > timeout.completed) & wait";
        let request = CommandRequest::shell(script, directory.path()).with_timeout_ms(500);

        let error = run(request).await.unwrap_err();
        assert!(error.to_string().contains("timed out"));
        assert!(directory.path().join("timeout.started").exists());

        sleep(Duration::from_millis(1_700)).await;
        assert!(
            !directory.path().join("timeout.completed").exists(),
            "the timed-out child must not survive to perform its delayed write"
        );
    }

    #[tokio::test]
    async fn dropping_the_run_future_kills_its_child_process_group() {
        let directory = tempdir().unwrap();
        let started = directory.path().join("dropped.started");
        let completed = directory.path().join("dropped.completed");
        let script = "(printf started > dropped.started; sleep 0.5; printf completed > dropped.completed) & wait";
        let request = CommandRequest::shell(script, directory.path()).with_timeout_ms(10_000);

        timeout(Duration::from_secs(2), async {
            tokio::select! {
                output = run(request) => panic!("the long-running command unexpectedly finished: {output:?}"),
                _ = wait_for_file(&started) => {},
            }
        })
        .await
        .expect("the command's child should start before the test deadline");

        sleep(Duration::from_millis(700)).await;
        assert!(
            !completed.exists(),
            "dropping the command future must stop its child before the delayed write"
        );
    }

    #[tokio::test]
    async fn invalid_working_directory_and_limits_do_not_spawn_the_command() {
        let directory = tempdir().unwrap();
        let marker = directory.path().join("invalid-request.spawned");
        let side_effect = format!("printf spawned > {}", shell_quote_path(&marker));

        let invalid_working_directory =
            CommandRequest::shell(side_effect.clone(), directory.path().join("missing"));
        assert!(run(invalid_working_directory).await.is_err());
        assert!(!marker.exists());

        let invalid_timeout =
            CommandRequest::shell(side_effect.clone(), directory.path()).with_timeout_ms(0);
        assert!(run(invalid_timeout).await.is_err());
        assert!(!marker.exists());

        let invalid_output_limit =
            CommandRequest::shell(side_effect, directory.path()).with_output_limit(0);
        assert!(run(invalid_output_limit).await.is_err());
        assert!(!marker.exists());
    }

    fn shell_quote_path(path: &Path) -> String {
        format!(
            "'{}'",
            path.to_string_lossy().replace(char::from(39), "'\\''")
        )
    }

    async fn wait_for_file(path: &Path) {
        while !path.exists() {
            sleep(Duration::from_millis(10)).await;
        }
    }
}
