
check_results=$(grep -rL "Copyright (C) 2024 Apple Inc. All rights reserved." "$@" | grep ".py$")

if [ -z "$check_results" ]
then
      exit 0
else
    echo "Python files are missing this license header:"
    echo "# For licensing see accompanying LICENSE file."
    echo "# Copyright (C) 2024 Apple Inc. All rights reserved."
    echo "$check_results"
    exit 1
fi
