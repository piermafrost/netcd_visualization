from matplotlib import pyplot as plt


# Put geographical (longitude-latitude) coordinate on axis
#---------------------------------------------------------


def geographic_ticks(ax=None, axis='both', **text_kwargs):

    ax = plt.gca() if ax is None else ax

    if axis=='x' or axis=='both':

        def xformatter(x, pos):
            if x==0:
                return '0°'
            elif x>0:
                return '{:g}°E'.format(x)
            elif x<0:
                return '{:g}°W'.format(-x)

        ax.xaxis.set_ticklabels([], **text_kwargs) # -> custom tick label without setting the text
        ax.xaxis.set_major_formatter(xformatter) # method to dynamically set the text
        ax.xaxis.set_minor_formatter('')

    if axis=='y' or axis=='both':

        def yformatter(y, pos):
            if y==0:
                return '0°'
            elif y>0:
                return '{:g}°N'.format(y)
            elif y<0:
                return '{:g}°S'.format(-y)

        ax.yaxis.set_ticklabels([], **text_kwargs) # -> custom tick label without setting the text
        ax.yaxis.set_major_formatter(yformatter) # method to dynamically set the text
        ax.yaxis.set_minor_formatter('')


#================================================#

