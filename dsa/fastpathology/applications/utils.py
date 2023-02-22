import fast


def force_run_exporters(pipeline):
    for name, po in pipeline.getProcessObjects().items():
        if po.getNameOfClass().endswith("Exporter"):
            pipeline.getProcessObject(name).run()
