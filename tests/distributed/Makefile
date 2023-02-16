make:
	g++ -std=gnu++0x -I${SYNAPSE_ROOT}/include \
        -I${SPECS_EXT_ROOT} -I${SPECS_ROOT} \
        -I${SYNAPSE_ROOT}/src/graph_compiler \
	-I${SYNAPSE_ROOT}/include/ -I${SYNAPSE_ROOT}/src/runtime \
       	-I${SYNAPSE_ROOT}/src/infra \
       	-I${SYNAPSE_ROOT}/src \
	-g -Wall -o hcl_bench hcl_bench.cpp \
        -L${SYNAPSE_RELEASE_BUILD}/lib/ -lSynapse -lpthread

clean:
	rm -f hcl_bench
