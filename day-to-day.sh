# 1. get the latest commits from the project's main
git fetch upstream

# 2. sync my fork's main
git push origin upstream/main:main

# 3. replay my commits on top of the latest upstream
git checkout wrp
git rebase upstream/main
#   on conflict: edit files, git add <file>, git rebase --continue
#   to bail out entirely: git rebase --abort

# 4. push
git push --force-with-lease origin wrp
