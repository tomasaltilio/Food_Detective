
from taxifare.data import get_data, clean_df, holdout, save_model_to_gcp
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline
from taxifare.metrics import compute_rmse
from taxifare.mlflow import MLFlowBase

import joblib


class Trainer(MLFlowBase):

    def __init__(self):
        super().__init__(
            "[FR] [Paris] [batch] taxifare + 1",
            "https://mlflow.lewagon.co")

    def retrieve_data(self):

        self.line_count = 1_000

        # get data
        df = get_data(self.line_count)
        df = clean_df(df)

        # holdout
        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)

    def mlflow_log_run(self):

        # create a mlflow training
        self.mlflow_create_run()

        # log params
        self.mlflow_log_param("model_name", "random_forest")
        self.mlflow_log_param("line_count", self.line_count)

        # push metrics to mlflow
        self.mlflow_log_metric("rmse", self.rmse)

    def evaluate_pipeline(self):

        # make prediction for metrics
        y_pred = self.pipeline.predict(self.X_test)

        # evaluate metrics
        self.rmse = compute_rmse(y_pred, self.y_test)

        print(f"rmse: {self.rmse}")

    def train(self):

        # step 1 : get data
        self.retrieve_data()

        # step 2 : create pipeline
        model = get_model()
        self.pipeline = get_pipeline(model)

        # step 3 : train
        self.pipeline.fit(self.X_train, self.y_train)

        # step 4 : evaluate perf
        self.evaluate_pipeline()

        # step 5 : save the trained model
        joblib.dump(self.pipeline, "model.joblib")

        # step 6 : upload model to gcp
        save_model_to_gcp()

        # step 7 : log run in mlflow
        self.mlflow_log_run()

        # return the pipeline to identify the hyperparams
        return self.pipeline


def main():

    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
