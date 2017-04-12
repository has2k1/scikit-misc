#!/bin/bash

# Auto-deploy sphinx documentation to gh-pages with Travis
# Credit: https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

set -o xtrace  # Print command traces before executing command

TOP_LEVEL=`git rev-parse --show-toplevel`
SOURCE_BRANCH="master"
TARGET_BRANCH="gh-pages"
HTML_DIRECTORY="${TOP_LEVEL}/doc/_build/html"
DOC_REPO_DIRECTORY="${TOP_LEVEL}/gh-pages"
ENCRYPTED_DEPLOY_KEY_FILE="${TOP_LEVEL}/tools/deploy_key.enc"
DEPLOY_KEY_FILE="${TOP_LEVEL}/tools/deploy_key"

# Get the deploy key by using Travis's stored variables to decrypt deploy_key.enc
ENCRYPTED_KEY_VAR="encrypted_${ENCRYPTION_LABEL}_key"
ENCRYPTED_IV_VAR="encrypted_${ENCRYPTION_LABEL}_iv"
ENCRYPTED_KEY=${!ENCRYPTED_KEY_VAR}
ENCRYPTED_IV=${!ENCRYPTED_IV_VAR}
openssl aes-256-cbc -K $ENCRYPTED_KEY -iv $ENCRYPTED_IV \
   -in $ENCRYPTED_DEPLOY_KEY_FILE -out $DEPLOY_KEY_FILE -d

chmod 600 $DEPLOY_KEY_FILE
eval `ssh-agent -s`
ssh-add $DEPLOY_KEY_FILE

# When HEAD is tagged and the tag indicates a releasable
# version (eg v1.2.3), then VERSION is that tag. Otherwise,
# it is an empty string.
release_re='[0-9]\+\.[0-9]\+\.[0-9]\+'
pre_re='\(\(a\|b\|rc\|alpha\|beta\)[0-9]*\)\?'

VERSION=$(git describe --always | grep "^v${release_re}${pre_re}$")

# Pull requests and commits to other branches should not deploy
# Deploy when master is tagged with a releasable version tag
if [[ -z $VERSION ]] || \
   [[ "$TRAVIS_PULL_REQUEST" != "false" ]] || \
   [[ "$TRAVIS_BRANCH" != "$SOURCE_BRANCH" ]]; then
    echo "Not deploying documentation"
    exit 0
fi

set -e # Exit with nonzero exit code if anything fails
# Save some useful information
REPO=`git config remote.origin.url`
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
SHA=`git rev-parse --verify HEAD`

# Clone the existing gh-pages for this repo into out/
# Create a new empty branch if gh-pages doesn't exist
# yet (should only happen on first deply)
git clone --depth=3 --branch=$TARGET_BRANCH $SSH_REPO $DOC_REPO_DIRECTORY || true
if [[ -d $DOC_REPO_DIRECTORY ]]; then
   pushd $DOC_REPO_DIRECTORY
else
   git clone -l -s -n . $DOC_REPO_DIRECTORY
   pushd $DOC_REPO_DIRECTORY
   git checkout $TARGET_BRANCH || git checkout --orphan $TARGET_BRANCH
   git reset --hard
fi

cp -a "$HTML_DIRECTORY/." ./

# Makes github-pages play nice with static content and assets
# contained in sub directories
if [[ ! -f .nojekyll ]]; then
   touch .nojekyll
fi

# Now let's go have some fun with the cloned repo
git config user.name "Travis CI"
git config user.email "travis@travis-ci.org"

# Commit the "changes", i.e. the new version.
# The delta will show diffs between new and old versions.
git add .

# If there are no changes to the compiled out (e.g. this is a README update) then just bail.
if [[ -z `git diff --cached --exit-code --shortstat` ]]; then
    echo "No changes to the output on this push; exiting."
    exit 0
fi

git commit -m "Documentation: ${VERSION}"

# Now that we're all set up, we can push.
git push $SSH_REPO $TARGET_BRANCH

popd
rm -rf $DOC_REPO_DIRECTORY
