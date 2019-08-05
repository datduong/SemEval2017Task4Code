
for name in train_bert_sentiment_file.txt test_bert_sentiment_file.txt dev_bert_sentiment_file.txt ; do 
  sed -i 's/\bthe\b//g' $name
  sed -i 's/\bThe\b//g' $name
done
