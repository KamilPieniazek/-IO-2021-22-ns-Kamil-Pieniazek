import pyswarms as ps
from pyswarms.backend.topology import Pyramid
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters import plot_contour
import matplotlib.pyplot as plt
from pyswarms.utils.functions.single_obj import sphere
from pyswarms.utils.plotters import plot_surface
from pyswarms.utils.plotters.formatters import Mesher
from IPython.display import Image


options = {'c1': 0.1, 'c2': 0.3, 'w':0.2}
my_topology = Pyramid(static=False)

optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                    options=options, topology=my_topology)

optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2,
                                    options=options)

stats = optimizer.optimize(fx.sphere, iters=100)
cost_history = optimizer.cost_history
pos_history = optimizer.pos_history
plot_cost_history(cost_history)
plot_contour(pos_history)
plt.show()

m = Mesher(func=fx.sphere)
cost, pos = optimizer.optimize(fx.sphere, iters=100)
pos_history_3d = m.compute_history_3d(optimizer.pos_history)
plot_surface(pos_history_3d)

animation = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
animation.save('plot0.gif', writer='imagemagick', fps=10)
Image(url='plot0.gif')
