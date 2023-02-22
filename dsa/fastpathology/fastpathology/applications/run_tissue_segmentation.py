import fast
import os
from utils import force_run_exporters


fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)

output = '/opt/pipelines/prediction.png'

pipeline = fast.Pipeline(
	'/opt/pipelines/tissue_segmentation.fpl',
	{
		'wsi': '/opt/pipelines/A05.svs',
		'output': output
	}
)

pipeline.parse()
force_run_exporters(pipeline)

print("Was export successful:", os.path.exists(output))
print("Result is saved at:", output)
