import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

# aliceblue, antiquewhite, aqua, aquamarine, azure,
# beige, bisque, black, blanchedalmond, blue,
# blueviolet, brown, burlywood, cadetblue,
css_colors = """chocolate, coral,
cornflowerblue, crimson, darkcyan, darkblue,
darkgoldenrod, darkgreen,
darkmagenta, darkolivegreen, darkorange,
darkorchid, darkred,
darkslateblue, darkslategrey,
darkturquoise, darkviolet, deeppink,
dimgray, dimgrey, dodgerblue, firebrick,
forestgreen, fuchsia,
goldenrod, gray, grey, green,
greenyellow, honeydew, hotpink, indianred, indigo,
ivory, lavender, lavenderblush, lawngreen,
lemonchiffon, lightblue, lightcoral, lightcyan,
lightgoldenrodyellow, lightgray, lightgrey,
lightgreen, lightpink, lightsalmon, lightseagreen,
lightskyblue, lightslategray, lightslategrey,
lightsteelblue, lightyellow, lime, limegreen,
linen, magenta, maroon, mediumaquamarine,
mediumblue, mediumorchid, mediumpurple,
mediumseagreen, mediumslateblue, mediumspringgreen,
mediumturquoise, mediumvioletred, midnightblue,
mintcream, mistyrose, moccasin, navajowhite, navy,
oldlace, olive, olivedrab, orange, orangered,
orchid, palegoldenrod, palegreen, paleturquoise,
palevioletred, papayawhip, peachpuff, peru, pink,
plum, powderblue, purple, red, rosybrown,
royalblue, rebeccapurple, saddlebrown, salmon,
sandybrown, seagreen, seashell, sienna, silver,
skyblue, slateblue, slategray, slategrey, snow,
springgreen, steelblue, tan, teal, thistle, tomato,
turquoise, violet, wheat, white, whitesmoke,
yellow""".replace('\n',' ').split(', ')

def make_plot_cols(numseries):
  if numseries <= 3:
    return ['blue', 'red', 'black']
  if numseries == 4:
    return ['blue', 'red', 'orange', 'black']
  if numseries == 5:
    return ['blue', 'navy', 'red', 'orange', 'black']
  if numseries == 6:
    return ['blue', 'navy', 'red', 'orange', 'gray', 'black']
  if numseries == 7:
    return ['blueviolet', 'blue', 'navy', 'red', 'orange', 'gray', 'black']
  if numseries > 7: #Thanks to the internet: http://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=numseries-1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    cvals = []
    for i in range(0, numseries):
      cvals.append(scalarMap.to_rgba(i))

    cvals = ['blue', 'red', 'orange', 'black']

    i = 0
    while len(cvals) < numseries:
        cvals.append(css_colors[i])
        i += 1
    return cvals
