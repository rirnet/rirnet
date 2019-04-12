cd cases_real
for i in *.wav ; do
  echo $i
  b=`basename $i .wav`
  lame --alt-preset insane $i $b.mp3
done
cd ..
