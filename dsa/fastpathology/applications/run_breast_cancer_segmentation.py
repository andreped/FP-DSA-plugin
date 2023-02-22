import fast
import os
from utils import force_run_exporters


fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)


# download models
fast.DataHub().download('breast-tumour-segmentation-model')

output = "/opt/pipelines/prediction"
model = "/root/FAST/datahub/"

pipeline = fast.Pipeline(
	'/opt/pipelines/breast_tumour_segmentation.fpl',
	{
		'wsi': '/opt/pipelines/A05.svs',
		'output': output,
		'model': model,
	}
)

pipeline.parse()
force_run_exporters(pipeline)

print("Was export successful:", os.path.exists(output + ".tiff"))
print("Result is saved at:", output)
