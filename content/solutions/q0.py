# Sanity-check snippet: just set a variable.
# tutolib.solution(0) runs this in the notebook's namespace, so the next cell
# `print(num)` works — proof that "Run it for me" really injects into the env.
num = 5
print("num is now", num)
