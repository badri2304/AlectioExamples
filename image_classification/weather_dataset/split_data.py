import splitfolders

splitfolders.ratio("data", output="output",
    seed=1337, ratio=(.8, .2), group_prefix=None, move=False)