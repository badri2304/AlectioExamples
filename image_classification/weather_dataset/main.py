import argparse
import yaml, json
from alectio_sdk.sdk import Pipeline
from processes import train, test, infer, getdatasetstate

with open("./config.yaml", "r") as stream:
    args = yaml.safe_load(stream)

# put the train/test/infer processes into the constructor
AlectioPipeline = Pipeline(
    name=args["exp_name"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
    token="4e97c472009143dfb11c30665e9a9436"
)

if __name__ == "__main__":
    AlectioPipeline()