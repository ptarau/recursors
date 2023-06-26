goal_prompter = dict(
    name='goal',
    and_p="""The plan so far is: "$context".
     In this context my goal is "$g." 
     Advise me how to achieve "$g", step by step, while ensuring each step is consistent with each other.
     Itemize your answer, one sentence per line.""",
    or_p="""The plan so far is: "$context".
    In this context my goal is "$g." 
    Suggest 2-3 mutually exclusive alternative ways to achieve "$g".
    Avoid starting your sentence with the word "Alternative"."""

)

causal_prompter = dict(
    name='causal',
    and_p="""We need causal explanations in this context: "$context"
        Generate 3-5 explanations of 2-4 words each for the causes of "$g".
        Itemize your answer, one reason for "$g" per line.
        No explanations needed, just the 2-4 words noun phrase, nothing else.
        Your answer should not contain ":" or "Cause".
        """,
    or_p="""We need causal explanation in this context: "$context"
        Generate 2-3 alternative explanations citing facts that might cause "$g".
        Itemize your answer, one noun phrase per line.
        No explanations needed, just the noun phrase, nothing else.
        Avoid starting your sentence with the word "Alternative".
        Your answer should not contain ":" .
        Your answer should avoid the word "Causes" and "causes" ."""
)

conseq_prompter = dict(
    name='conseq',
    and_p="""We need to predict consequences this context: "$context"
        Generate 3-5 noun phrases of 2-4 words each detailing consequences of "$g".
        Itemize your answer, one consequence of "$g" per line.
        No explanations needed, just the noun phrase, nothing else.
        Avoid using the words "Noun phrases" in your answer.
        Your answer should not contain ":".""",
    or_p="""We need to predict consequences of "$g" inthis context: "$context"
        Generate 2-3 alternative predictions citing facts that are likely to be consequences of "$g".
        Itemize your answer, one noun phrase per line.
         No explanations are needed, just the noun phrase, nothing else.
        Avoid starting your sentence with the word "Alternative".
        Your answer should not contain ":"."""
)

sci_prompter = dict(
    name='sci',
    and_p="""The task we are exploring is: "$context"
        Generate 3-5 noun phrases of 2-4 words each that occur as keyphrases only
        in scientific papers bout "$g".
        Itemize your answer, one noun phrase per line.
        No explanations needed, just the noun phrase, nothing else.
        """,
    or_p="""The topic we are exploring is: "$context"
        Generate 2-3 noun phrases describing details of "$g".
        Itemize your answer, one noun phrase per line.
        No explanations needed, just the noun phrase, nothing else.
        """
)

recommendation_prompter_strict = dict(
    name='rec_strict',
    and_p="""The recommandations so far are: "$context".
     In this context I really liked "$g." 
     Suggest me 2-3 related ones.
     Itemize your answer, one recommendation per line.
     Just give me each title one its line, no comments or summaries, no text before and after the title.""",
    or_p="""""The recommandations so far are: "$context".
     In this context I am considering "$g".
     Suggest 2-3 distinct alternative recommandations insted of "$g".
     Just give me each title on its line, no comments or summaries, no text before and after the title.
     Avoid starting your sentence with the word "Sure".
     """
)

recommendation_prompter = dict(
    name='rec',
    and_p="""The recommandations so far are: "$context".
     In this context I really liked "$g." 
     Suggest me 2-3 related ones that go well together.
     Itemize your answer, one recommendation per line.
     Just give me each title on its line, no comments or summaries.""",
    or_p="""""The recommandations so far are: "$context".
     In this context I am considering "$g".
     Suggest 2-3 distinct alternative recommandations insted of "$g".
     Just give each title on its line, no comments or summaries.
     Avoid starting your sentence with the word "Sure" and avoid having commas in your answer.
     """
)


decision_prompter = dict(
    name='oracle',
    decider_p="""
    You play the role of an oracle that decides if "$g" is similar to "$context". This means that someone 
    who is familiar with "$context"  might be also be interested in "$g".
    Your answer should de "True" or "False" expressing strong agreement or some disagreement with the relevance of "$g".
    No explanations or comments are needed, just one of the two words "True" or "False"
    """
)

relevance_prompter = dict(
    name='oracle',
    decider_p="""
    You play the role of an oracle that decides if "$g" is semantically close and strongly relevant for "$context".
    Your answer should de "True" or "False" expressing strong agreement or some disagreement with the relevance of "$g".
    No explanations or comments are needed, just one of the two words "True" or "False"
    """
)

ratings_prompter = dict(
    name='oracle',
    rater_p="""
    On a scale from 0 to 100, rate how relevant and semantically close "$g" is to "$context".
    Return your answer in the form: rating | explanation.
    """
)

sci_abstract_maker = dict(
    name='sci_abstract_maker',
    writer_p="""
    Suggest a title and an abstract for a scientific paper about "$g" that has the following keywords: $context.
    """
)
