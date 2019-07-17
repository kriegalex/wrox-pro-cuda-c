APPS=query_openacc_devices reduction-demo atomic-demo access-patterns loop-schedule

all: ${APPS}

%: %.c
	pgc++ -acc -Minfo=accel $< -o $@ -lcudadevice

clean:
	rm -f ${APPS}
