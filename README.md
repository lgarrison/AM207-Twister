1950-2014_torn.csv is the original file from http://www.spc.noaa.gov/wcm/#data

1950-2014_torn.warnings.csv has 4 additional columns:
- datetime_gmt: a numpy datetime64 object that encode the date and time of the touchdown.
- has_warning: 1 if the tornado had a warning issued, 0 otherwise
- has_watch: 1 if the tornado had a watch issued, 0 otherwise
- warning_time: the number of minutes between the issuance of the first warning and the tornado touchdown. Negative = warning issued after touchdown.

We consider a watch/warning to be issued for a given tornado if the tornado passes through a county that had an active warning at the time, where "active" is within an hour or two.  See the notebook for the full explanation.
There is warnings data for 1986+, and watches data for 2006+.
    
tornadoes_pipeline.ipynb is the notebook used to join the warnings and tornado data

The watches+warnings data from https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml is huge and not included here.
