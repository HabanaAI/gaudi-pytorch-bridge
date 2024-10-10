# Shell commands to combine FW and Synapse tracing
cat .local.synapse_log.json     >  combined.json
sed '1d;$d' fw_event_trace.json >> combined.json
echo "]"                        >> combined.json
