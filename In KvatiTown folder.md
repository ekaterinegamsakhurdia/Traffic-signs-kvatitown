In KvatiTown folder:



git remote -v



Add new repo:



git remote add traffic https://github.com/ekaterinegamsakhurdia/Traffic-signs-kvatitown.git



Commit your current changes:



git add .

git commit -m "Add traffic signs final project"



Push only to the new repo:



git push -u traffic main



If your branch is master, use:



git push -u traffic master



Check branch name with:



git branch



After this:



old repo stays as origin

new repo is traffic

your changes are saved to the new traffic-signs repo

nothing is pushed to old repo unless you run git push origin ...



For future pushes to the new repo:



git push traffic main

