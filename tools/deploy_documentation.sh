#!/bin/bash

# Auto-deploy sphinx documentation with Github Actions
# Credit: https://gist.github.com/domenic/ec8b0fc8ab45f39403dd
set -o xtrace       # Print command traces before executing command
set -e              # Exit with nonzero exit code if anything fails
echo $BASH_VERSION  # For debugging

# Setup variables
COMMIT_AUTHOR_NAME="Github Actions"
COMMIT_AUTHOR_EMAIL="github-actions@github.com"
release_re='[0-9]\+\.[0-9]\+\.[0-9]\+'
pre_re='\(\(a\|b\|rc\|alpha\|beta\)[0-9]*\)\?'
VERSION=$(echo $SOURCE_TAG | grep "^v${release_re}${pre_re}$") || VERSION="unknown"
COMMIT_MSG="Documentation: ${VERSION}"

# Pull requests and commits to other branches should not deploy
# Deploy when master is tagged with a releasable version tag
if [[ $VERSION == "unknown" ]] || \
   [[ "$SOURCE_BRANCH" != "master" ]]; then
    echo "Not deploying documentation"
    exit 0
fi

# Copy documentation
rm -rf .
cp -a "$HTML_DIR/." ./

# Configure commit information
git config user.name "$COMMIT_AUTHOR_NAME  [GA Deploy Doc.]"
git config user.email "$COMMIT_AUTHOR_EMAIL"

# Commit the "changes", i.e. the new version.
# The delta will show diffs between new and old versions.
git add .

if [[ -z `git diff --cached --exit-code --shortstat` ]]; then
  echo "No changes to the output on this push; exiting."
  exit 0
fi

echo "SUCCESS"
# git commit -m "$COMMIT_MSG"
# git push
