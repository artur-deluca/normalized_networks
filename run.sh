for k in 2 3 5 
do
	rm ./here/$k -r
	python train --parametric --normalize --epochs 600 --layers $k --save ./here/$k --seed 11
	python train --epochs 600 --layers $k --save ./here/$k --seed 11 
	for i in 0.005 0.05 0.5 1.5 3 5 10
	do
		python train --parametric --alpha $i --epochs 600 --layers $k  --save ./here/$k --seed 11
	done
done
