neuralnet: neuralNetwork.cu main.cu
	nvcc -o nn *.cu -lm -I .
	echo "0 0"
	./nn 0 0
	echo "1 0"
	./nn 1 0
	echo "0 1"
	./nn 0 1
	echo "1 1"
	./nn 1 1

