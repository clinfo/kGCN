#kgcn train --config config_cnn.json  --verbose --cpu

kgcn visualize --config config_cnn.pos.json  --cpu --verbose --ig_label_target 1

for f in `ls viz_cnn/*.jbl`
do
	gcnv -i $f
done
