
for wav in TSur/*.wav; do
		base=`basename ${wav} .wav`;
		echo "wav2raw +s wav/$base.wav";
		sox TSur/$base.wav -r 22050 down/$base.wav;
done

