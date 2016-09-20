all:
	python2.7 build.py build_ext --inplace

clean:
	rm -f *.so
	rm -f *.pyc
	rm -rf build
	rm -rf tmp
	rm -rf dist
	rm -f *.log
	rm -rf paralution_wrapper.egg-info

install:
	python2.7 build.py install --user
	
test:
	python2.7 csr_test.py


