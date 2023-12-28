## DeepQA is a [DeepLLM](https://github.com/ptarau/recursors/tree/main)-based application that explores recursively the "mind-stream" of an LLM via a tree of self-generated follow-up questions.

Use 

```./deep_qa_app.sh``` 

to start a streamlit based interactive app exploring a tree of follow-up questions together with their answers up to a given depth.

If you fetch the project and run it locally, you can import the generated Definite Clause Grammar as part of a Prolog program. It replicates symbolically the equivalent of the "mind-stream" extracted from the LLM interaction, with possible uses of the encapsulated knowledge in Logic Programming applications.

The synthesized grammar is designed to generate a finite language (by carefully detecting follow-up questions that would induce loops), We also ensure that paths in the question-answer tree are free of repeated answers, which get collected as well, together with questions left open as a result of reaching the user-set depth limit.

You can use DeepQA to quickly assess the *strength* of an LLM before committing to it.

For instance, when used with a much weaker than GPTx local LLM (enabled with Vicuna 7B by default) you will see shorter, more out of focus results, with a lot of repeated questions and answers collected by DeepQA in corresponding bins.

Enjoy,

Paul Tarau

