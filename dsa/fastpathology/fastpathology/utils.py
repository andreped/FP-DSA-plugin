import fast


def force_run_exporters(pipeline):
    for name, po in pipeline.getProcessObjects().items():
        if po.getNameOfClass().endswith("Exporter"):
            pipeline.getProcessObject(name).run()


def enable_fast_verbosity():
    fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)


def download_models(model_name):
    model_name = model_name.replace("_", "_") + "-model"
    fast.DataHub().download(model_name)


def run_pipeline(fpl, input_, output, model):
    pipeline = fast.Pipeline(fpl, {"input": input_, "output": output, "model": model})
    pipeline.parse()
    force_run_exporters(pipeline)
