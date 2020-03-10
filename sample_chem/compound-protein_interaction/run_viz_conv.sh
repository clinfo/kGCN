echo -n "" > run_viz_conv.sh.list
for f in `ls ./viz_st0/*.jbl`
do
    echo "gcnv -i $f" >> run_viz_conv.sh.list
done
for f in `ls ./viz_st1/*.jbl`
do
    echo "gcnv -i $f" >> run_viz_conv.sh.list
done
for f in `ls ./viz_st2/*.jbl`
do
    echo "gcnv -i $f" >> run_viz_conv.sh.list
done
for f in `ls ./viz_st3/*.jbl`
do
    echo "gcnv -i $f" >> run_viz_conv.sh.list
done
#
#xargs -a ./run_viz_conv.sh.list -P 64 -d '\n' -I {} sh -c {}

