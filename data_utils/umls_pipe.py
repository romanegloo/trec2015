#!usr/bin/evn python3

from __future__ import print_function
import nltk
from nltk.tokenize import sent_tokenize

example_str = """Evidence is increasing that oxygen debt and its metabolic
correlates are important quantifiers of the severity of hemorrhagic and
post-traumatic shock and and may serve as useful guides in the treatment of
these conditions. The aim of this review is to demonstrate the similarity
between experimental oxygen debt in animals and human hemorrhage/post-tra
umatic conditions, and to examine metabolic oxygen debt correlates, namely base
deficit and lactate, as indices of shock severity and adequacy of volume
resuscitation. Relevant studies in the medical literature were identified using
Medline and Cochrane Library searches. Findings in both experimental animals
(dog/pig) and humans suggest that oxygen debt or its metabolic
correlates may be more useful quantifiers of hemorrhagic shock than
estimates of blood loss, volume replacement, blood pressure, or
heart rate. This is evidenced by the oxygen debt/probability of death curve
s for the animals, and by the consistency of lethal dose (LD)<sub>25,50
</sub>points for base deficit across all three species. Quantifying
human post-traumatic shock based on base deficit and adjusting for
Glasgow Coma Scale score, prothrombin time, Injury Severity Score
and age is demonstrated to be superior to anatomic injury severity
alone or in combination with Trauma and Injury Severity Score.
The data examined in this review indicate that estimates of oxygen debt and
its metabolic correlates should be included in studies of experimental shock and
 in the management of patients suffering from hemorrhagic shock."""


sentences = sent_tokenize(example_str)
sample = False
print(len(sentences))

tokenized = nltk.word_tokenize(sentences[1])
tagged = nltk.pos_tag(tokenized)
ner = nltk.ne_chunk(tagged)
if not sample:
    print(ner)
    sample = True


