for wav in SNeu/*.wav; do
    base=`basename ${wav} .wav`;
    echo "wav2raw +s wav/$base.wav > raw/${base}.raw";
    sox SNeu/$base.wav -b 16 --rate 22050 SNeu/$base.wav;
done
