Using nnitp
===========

Starting nnitp
--------------

Start nnitp by entering this command::

  nnitp

User interface overview
-----------------------

The user interface has two parts:

- A messages pane on the top, and 
- A collection of tabs on the bottom.

The tabs are:

- The `Model` tab, controller the model, and the explanation parameters, and
- The `Images` tab, which displays the image grid.

The Model tab
^^^^^^^^^^^^^

The `Model` tab contains:

- A drop-down box for selecting the model name
- A data set selection box
- Various hyper-parameters of the interpolation algorithm:

  - `alpha`: the precision parameter (float 0--1.0)
  - `gamma`: the recall parameter (float 0--1.0)
  - `mu`: the recall reduction parameter (float 0--1.0)
  - `size` : the sample size  (int > 0)
  - `ensemble size`: number of interpolants in ensemble (int > 0)
  - `layers`: list of layers to use in interpolants

The hyper-parameters can be set in the model file, and modified in the
UI.  The `layers` parameter is presented as a list selector which
provides a list of available layers in its left panel and shows the
list of selected layers in its right panel. To move a layer from the
available list to the selected list, select it by clicking and then
click the `>` button. To move a selected layer back to the available
list, select it by clicking and then click the `<` button.

The Images tab
^^^^^^^^^^^^^^

The `Images` tab shows the current state of the explanation process.
It contains:

- A `category` selection (int)
- A `restrict` check-box
- A `back` button
- Indicators for `predicate` and `fraction`
- A percentile selection (float 0--100)
- The image grid.
  
Initially, the image grid shows a subset of the images from the
selected data set (either `training` or `test`) that are predicted by
the model be assigned to the selected `category`. The user may select
an image to explain the model's prediction.

Once an image is selected, the interface enters the `normal` state. In
the normal state, a grid of images is displayed. The `key` image in the upper
left corner of the grid is always the selected image. By hovering the
cursor over an image, the `predicate` satisfied by that image is
displayed, as well as the `fraction` of images in the chosen data set
that satisfy the predicate. Upon initially selecting an image, its
predicate is `is_max(N)`, where `N` is the selected category,
indicated that `N` is the `arg max` of the output units. The
`predicate` indicator shows both the predicate and the `layer` at
which the predicate is evaluated.

To explain why an image satisfies its predicate, the user clicks the
image. Nnitp will then computes an interpolant predicate and the
nearest preceding layer that has been selected for interpolation. For
example, if the images predicate is at layer 25, and the layers
selection for interpolation are 6 and 14, then an interpolant is
computed at layer 14. The interpolant formula is printed in the
message pane, along with its precision and recall over the training and
test sets. In the interpolant formulas, `v(x,y,...)` stands for the
activation of the unit with tensor coordinates `x,y,...` in the given
layer.

Once an interpolant is computed, the interface moves to a new state,
with a new image grid. Below the key image appear the `cones` of each
conjunct of the interpolant. The cone of a predicate is the region of
the image upon which depends. If the interpolant is a logical `and` of
`k` constraints on individual units, then there are `k` cone regions,
each having one unit constraint as its predicate, each show the the
subset of the image pixels that influence that constraint. Generally
speaking, for a sequential net using convolutional and pooling layers,
units nearer to the output will have larger cones.

To show the center position in the key image of each region, and digit
is displayed, from 0 to `k-1`.

To the right of the key image in the grid are shown `comparison`
images. These are images from the dataset that also satisfy the
predicate of the key image (usually, the computed interpolant). As
with the key image, below each comparison image are shown the cone
regions for each of the predicate conjuncts, and the cone locations
are indicated with digits. The comparison images can be used to help
interpret the predicate. Checking the `restrict` box restrict the
comparison images to those that are predicted to be in the chose
`category`.

When an image in the grid is right-clicked, and the `Examples` context
menu item is selected, the comparison predicate is changed to the
predicate of the selected image.  If `Counterexamples` is selected,
the `negation` of the image's predicate is used. Thus the comparison
images will be images that do not satisfy the image's
predicate. Examples and counterexamples can be used to help interpret
particular conjuncts of an interpolant.

Before selecting `Examples` or `Counterexamples` a number `n` in the
range 0--100 can be entered in the `percentile` box. In this case, the
comparison images will be those in the `n-th` percentile of the
selected predicate. For example, of the predicate is `v(x,y,z) >= 0.5`
then the comparison images will be those in the upper `n-th`
percentile of `v(x,y,z)`. This can be used to find images in which the
given unit is strongly or weakly activated.

