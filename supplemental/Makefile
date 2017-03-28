
k80_files = $(wildcard ../results/K80/TLB-Test-*.csv)
p100_files = $(wildcard ../results/P100/TLB-Test-*.csv)

TARGETS = K80	P100

all : k80 p100

k80 : $(k80_files)

p100 : $(p100_files)

$(k80_files) :
	@echo "input: $(@)"
	./tlb-benchmark-plot.r $@ > $(@).log 2>&1

$(p100_files) :
	@echo "input: $(@)"
	./tlb-benchmark-plot.r $@ > $(@).log 2>&1

clean :
	rm -vf ../results/*/*.pdf
	rm -vf ../results/*/*.log
