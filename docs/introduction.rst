Introduction
============

Statistical machine learning models, most notably deep neural
networks, have recently proven capable of accomplishing a remarkable
variety of tasks, such as image classification, speech recognition,
and visual question answering, with a degree of precision formerly
only achievable by humans. There has been increasing concern, however,
about the opacity of these models. That is, while the models may
produce accurate predictions, they do not produce explanations of
these predictions that can be understood and validated by humans.
This opacity can have negative consequences when models are applied in
the real world. From a technical point of view, it creates an obstacle
to improving models and methods.

Nnitp is tool that attempts to produce *explanations* for inferences
made by neural models that are at the same time *quantitatively
precise* and understandable to humans.  It takes as its starting point
a notion of explanation called a *Craig interpolant* that is widely
used in automated reasoning.  Given a logical premise `A` that implies
a conclusion `B`, an *interpolant* for this implication is an
intermediate predicate `I`, such that:

1. `A` implies `I`,
2. `I` implies `B`, and
3. `I` is expressed only using variables that are in common to `A` and `B`.
   
Another way to say this is that an interpolant is an intermediate fact
over intermediate variables.  This is crucial to the view of
interpolants as explanations.  From an intuitive point of view, for
`A` to explain a fact to `B`, it must speak the language of `B`,
abstracting away concepts irrelevant to `B`. The key properties of an
interpolant are that it should be *simple*, in order to avoid
over-fitting, and it should use the right vocabulary, in order to
abstract away irrelevant concepts. 

How, then can we transfer this notion of "intermediate fact over
intermediate variables" to a *statistical* inference made by a neural
net? Nnitp does this by replacing logical proof with a kind of naive
Bayesian "proof".  An inference in this proof will be of the form
:math:`P(B | A) \geq \alpha` where :math:`\alpha` is a desired degree of
certainty. In the statistical setting, however, our premise `A` and
conclusion `B` need not have any variables in common, since premise
and conclusion are connected by an underlying probability
distribution. Instead of using the common vocabulary as our notion of
"intermediate", we will simply choose some set of variables `V` that
we consider to be in some way intermediate between premise `A` and
conclusion `B` in the chain of inference. We then define a (naive)
Bayesian interpolant to be a predicate `I` over the intermediate
variables `V` such that

1. :math:`P(I | A) \geq \alpha` and
2. :math:`P(B | I) \geq \alpha`.

That is, if we observe `A`, then `I` is probably true, and if we
observe `I`, then `B` is probably true. This is a "naive" proof
because it implies that :math:`P(A | B) \geq \alpha^2`, but only under
the unwarranted assumption that `A` and `B` are independent given `I`.

For Nnitp, the premise `A` is an input presented to the network, the
conclusion `B` is the prediction made by the network, and the
intermediate vocabulary `V` represents the activation of the network
at some hidden layer. Despite possibly unjustified assumptions, Nnitp
can produce explanations that are both understandable to humans, and
remarkably precise.

For more information on how Nnitp computes interpolants, see `this
paper <https://arxiv.org/abs/2004.04198>`_.



