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
VERSION=$(echo $SOURCE_TAG | grep "^v${release_re}${pre_re}$") || VERSION=""
RELEASE_VERSION=$(echo $SOURCE_TAG | grep "^v${release_re}$") || RELEASE_VERSION=""
DEST_DIR=""
NEW_RELEASE="No"

# dev, latest, stable, v1.0.0
if [[ "$SOURCE_BRANCH" == "main" ]]; then
    DEST_DIR="latest"
elif [[ "$SOURCE_BRANCH" == "dev" ]]; then
  DEST_DIR="dev"
elif [[ "${SOURCE_BRANCH:0:11}" == "refs/tags/v" ]]; then
  if [[ "$RELEASE_VERSION" ]]; then
    DEST_DIR="$RELEASE_VERSION"
  fi
fi

# A release (tag) without a corresponding directory entry means we
# are seeing it for the first time.
if [[ "$DEST_DIR" == "$RELEASE_VERSION" ]] && [[ ! -d "$DEST_DIR" ]]; then
  NEW_RELEASE="Yes"
fi

# Do not deploy if destination directory has been created at this point.
# If the destination directory exists clear it out
# Otherwise we create it
if [[ -z "$DEST_DIR" ]]; then
  echo "Nothing to deploy."; exit 0
elif [[ -d "$DEST_DIR" ]]; then
  rm -f "$DEST_DIR/*"
else
  mkdir -p $DEST_DIR
fi

# Copy documentation
touch .nojekyll
cp -a "$HTML_DIR/." $DEST_DIR

# A new release becomes the stable version
# and also becomes the latest
if [[ "$NEW_RELEASE" == "Yes" ]]; then
  rm stable

  ln -s "$DEST_DIR" stable

  rm -rf latest
  ln -sf $DEST_DIR latest

  # For debugging
  ls -la
fi

COMMIT_MSG="Documentation: ${DEST_DIR}"

# Configure commit information
git config user.name "$COMMIT_AUTHOR_NAME"
git config user.email "$COMMIT_AUTHOR_EMAIL"

# Commit the "changes", i.e. the new version.
# The delta will show diffs between new and old versions.
git add "$DEST_DIR"
git add stable
git add latest

changes=$(git diff --cached --shortstat)
if [[ -z "$changes" ]]; then
  echo "No changes to the output on this push; exiting."
  exit 0
fi

git commit -m "$COMMIT_MSG"
git push
