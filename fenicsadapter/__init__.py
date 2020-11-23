import warnings
warnings.warn("The package name fenicsadapter is deprecated. Please use fenicsprecice instead. Example: Use 'from fenicsprecice import Adapter', not 'from fenicsadapter import Adapter'. Using fenicsadapter will lead to an error in future releases.")

# implemented following https://stackoverflow.com/a/24324577/5158031
import sys
import fenicsprecice

sys.modules[__name__] = sys.modules['fenicsprecice']
