# Load the pod's environment in an ssh shell: RunPod secrets are only in the environment of the container's first process.
while IFS= read -r -d '' v; do
  case "$v" in
    HF_TOKEN=*|GITHUB_API=*|AIRON_API_KEY=*|RUNPOD_API_KEY=*|RUNPOD_POD_ID=*) export "$v" ;;
  esac
done < /proc/1/environ
