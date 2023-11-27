import netCDF4
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import animation as anim # => use of it not yet implemented
from graphic_functions import geographic_ticks
from geographic_functions import cell_area

# For shaedit:
from matplotlib import widgets as mp_wdgt
from matplotlib.path import Path as mp_Path
from matplotlib.gridspec import GridSpec




# TO DO NEXT:
#
# Put axis name
# Add geographic grid?




#::::::::::::::::::::::::#
#:  TECHNICAL FUNCTIONS :#
#::::::::::::::::::::::::#

def remove_all(listrm, value):
    remains = True
    while remains:
        try:
            k = listrm.index(value)
            del(listrm[k])
        except ValueError:
            remains = False

    return listrm


############################
##  PRELIMINARY FUNCTIONS ##
############################


def get_axis_and_figure(fig=None, ax=None, overlay=False):
    '''
    Create figure and/or axis if they are given as 'None',
    or let them unchanged if not.
    '''

    if fig is None and ax is None:
        if overlay:
            fig, ax = plt.gcf(), plt.gca()
        else:
            fig, ax = plt.subplots()
    elif ax is None:
        if overlay:
            ax = plt.gca()
        else:
            ax = fig.subplots()
    else: # fig is None
        fig = ax.get_figure()

    return fig, ax




##########################################################
##  PRELIMINARY CLASS (numpy array with physical unit)  ##
##########################################################


LONGITUDE_LEGAL_VARNAMES = ['lon', 'longi', 'longitude', 'nlon']
LATITUDE_LEGAL_VARNAMES  = ['lat', 'lati', 'latitude', 'nlat']
LONGITUDE_LEGAL_UNITS = ['degrees_east', 'degrees east', 'degrees_e', 'degrees e', 'degree_east', 'degree east',
                         'degree_e', 'degree e', 'deg_east', 'deg east', 'deg_e', 'deg e', '°e', 
                         'degrees_west', 'degrees west', 'degrees_w', 'degrees w', 'degree_west', 'degree west',
                         'degree_w', 'degree w', 'deg_west', 'deg west', 'deg_w', 'deg w', '°w', 
                         'degrees', 'degree', 'deg', '°']
LATITUDE_LEGAL_UNITS  = ['degrees_north', 'degrees north', 'degrees_n', 'degrees n', 'degree_north', 'degree north',
                         'degree_n', 'degree n', 'deg_north', 'deg north', 'deg_n', 'deg n', '°n',
                         'degrees_south', 'degrees south', 'degrees_s', 'degrees s', 'degree_south', 'degree south',
                         'degree_s', 'degree s', 'deg_south', 'deg south', 'deg_s', 'deg s', '°s',
                         'degrees', 'degree', 'deg', '°']


class physical_array():


#    __precision = { # define the relative precision to determine is fields are equal or different
#                    'int':      np.array(0, dtype='int'),
#                    'int8':     np.array(0, dtype='int8'),
#                    'int16':    np.array(0, dtype='int16'),
#                    'int32':    np.array(0, dtype='int32'),
#                    'int64':    np.array(0, dtype='int64'),
#                    'float16':  np.array(1e-3, dtype='float16'),
#                    'float':    np.array(1e-7, dtype='float'),
#                    'float32':  np.array(1e-14, dtype='float32'),
#                    'double':   np.array(1e-14, dtype='double'),
#                    'float64':  np.array(1e-14, dtype='float64'),
#                    'float128': np.array(1e-29, dtype='float128')
#                   }


    def __init__(self, array, units):
        # 'array' is meant to be an numpy array (will be converted into it),
        # 'units' must be a string
        self.value = np.array(array)
        self.units = units
        self.shape = self.value.shape


    # Type check: test if 2 objects have the same type:
    def _check_type(self, other, raiseerror=True):
        if (type(self) != type(other)):
            if raiseerror:
                raise TypeError('Second argument must be a "ncGriddedVar" object')
            else:
                return False
        else:
            return True


    # Units test: check if 2 objects "ncGriddedVar" have the same units
    def _check_units(self, other, raiseerror=True):
        if (self.units != other.units):
            if raiseerror:
                raise ValueError('Variables do not have the same units')
            else:
                return False
        else:
            return True


    # Methods common to all children classes (can be redefined in children classes) used notably for equality test
    def _get_shape(self):
        return self.shape

    def _get_data(self):
        return self.value # Note for children classes: a numpy array is expected to be returned (see __eq__ method)


    # Re-define the equlatity test: same type, same units, same shape and same values
    def __eq__(self, other):
        if self._check_type(other, raiseerror=False):
            if self._check_units(other, raiseerror=False) and (self._get_shape() == other._get_shape()):
                return np.equal(self._get_data(), other._get_data()).all()
            else:
                return False

        else:
            return False




################################################################
##  MAIN CLASSES  ('ncDImenison', 'ncGriddedVar' and 'load')  ##
################################################################


class ncDimension(physical_array):



    def __init__(self, name, from_netCDF4_Dataset=None,
                 positions=None, bounds=None, bounds_1D=None, units='', axis='', positive='up'):

        self._nc4_dataset     = from_netCDF4_Dataset
        self.name             = name
        # Note: those attribute will be overridden if dimension is from a netCDF dataset
        self.positions        = positions
        self.bounds           = bounds
        self.bounds_1D        = bounds_1D
        self.units            = units
        self.axis             = axis
        self.positive         = positive

        if self._nc4_dataset is None:

            if (positions is None  and  bounds is None  and  bounds_1D is None):
                raise ValueError('ERROR: not enough information to create a new dimension')

            self._nc4_parent          = None
            self._nc4_parent_variable = None
            
            if units=='':
                print('WARNING: new dimension "'+name+'" assumed to be dimensionless')

        else:

            self._nc4_parent              = self._nc4_dataset.dimensions[name]
            self.name                     = self._nc4_parent.name
            self.size                     = self._nc4_parent.size

            if name in self._nc4_dataset.variables:
                self._nc4_parent_variable = self._nc4_dataset.variables[name]
                self.positions            = self._nc4_parent_variable[:]
                self.bounds               = self._get_bounds_from_ncDataset() # get 'bounds' attribute, or return None
                self.bounds_1D            = None
                if hasattr(self._nc4_parent_variable, 'units'):
                    self.units            = self._nc4_parent_variable.units
                else:
                    print('WARNING: attribute "units" not found for dimension variable "'+name+'". Assume dimensionless.')
                    self.units            = ''

                if hasattr(self._nc4_parent_variable, 'axis'):
                    self.axis             = self._nc4_parent_variable.axis
                else:
                    self.axis             = ''

            else:
                print('WARNING: Dimension "'+name+'" does not have corresponding variable. Use 1,...,N instead')
                self._nc4_parent_variable = None
                self.positions            = np.arange(1, self.size+1, 1)
                self.bounds               = None
                self.bounds_1D            = None

            if hasattr(self._nc4_parent, 'positive'):
                self.positive             = self._nc4_parent.positive.lower()
            else:
                self.positive             = 'up'


        # Compute boundaries or positions if not found nor documented:
        if self.bounds is None and self.bounds_1D is None:

            if self._nc4_dataset is None:
                print('WARNING: cell boundaries in dimension "'+self.name+'" not documented')
            else:
                print('WARNING: cell boundaries in dimension "'+self.name+'" not found.')

            print('    => automatic computation assuming middle between cell positions.')
            size = self.positions.size
            self.bounds_1D       = np.zeros((size+1), dtype=float)
            self.bounds_1D[1:-1] = (self.positions[:-1] + self.positions[1:]) / 2
            self.bounds_1D[0]    = 2*self.positions[0] - self.bounds_1D[1]
            self.bounds_1D[-1]   = 2*self.positions[-1] - self.bounds_1D[-2]
            self.bounds          = np.zeros((size,2), dtype=float)
            self.bounds[:,0]     = self.bounds_1D[:-1]
            self.bounds[:,1]     = self.bounds_1D[1:]

        elif self.bounds is None:

            size = self.bounds_1D.size-1
            self.bounds          = np.zeros((size,2), dtype=float)
            self.bounds[:,0]     = self.bounds_1D[:-1]
            self.bounds[:,1]     = self.bounds_1D[1:]

        elif self.bounds_1D is None:

            size = self.bounds.shape[0]
            self.bounds_1D       = np.zeros((size+1), dtype=float)
            self.bounds_1D[0]    = self.bounds[0,0]
            self.bounds_1D[-1]   = self.bounds[-1,1]
            self.bounds_1D[1:-1] = (self.bounds[1:,0] + self.bounds[:-1,1]) / 2

        if self.positions is None:

            print('WARNING: positions in dimension "'+name+'" not documented')
            print('    => automatic computation assuming middle between cell bounds.')
            self.positions       = self.bounds.mean(1)

        self.size          = self.positions.size

        self._is_longitude = (self.name.lower() in LONGITUDE_LEGAL_VARNAMES)
        self._is_latitude  = (self.name.lower() in LATITUDE_LEGAL_VARNAMES)


    # Special method, used in __init__
    def _get_bounds_from_ncDataset(self):
        for bnd in ['boundaries', 'boundary', 'bounds', 'bound', 'bnds', 'bnd']:
            bndname = self.name+'_'+bnd
            if bndname in self._nc4_dataset.variables:
                bounds_var = self._nc4_dataset.variables[bndname]
                if bounds_var.get_dims()[1].name == self.name:
                    return bounds_var[:,:].transpose()
                else:
                    return bounds_var[:,:]

        return None

    # Method to correct longitude or latitude bounds
    def _check_bounds(self):

        # get dimension orientation:
        if self.positions[-1] > self.positions[0]:
            i0, i1 = 0, -1
        else:
            i0, i1 = -1, 0

        if self._is_latitude:
            # Avoid exceedance of North or South pole:
            self.bounds_1D[i0] = max(self.bounds_1D[i0], -90)
            self.bounds_1D[i1] = min(self.bounds_1D[i1], 90)
        elif self._is_longitude:
            # Avoid overlapping in periodic boundary conditions:
            split = (self.bounds_1D[i0] + self.bounds_1D[i1]-360) / 2
            self.bounds_1D[i0] = split
            self.bounds_1D[i1] = split+360

        self.bounds[0,0] = self.bounds_1D[0]
        self.bounds[-1,1] = self.bounds_1D[-1]


    # Re-define common methods for equality check
    def _get_shape(self):
        return (self.size,)

    def _get_data(self):
        return self.positions
    

    # Add name condition to parent class equality test
    def __eq__(self, other):
        passed = super().__eq__(other)
        if passed:
            passed = (self.name == other.name)

        return passed


    # Display dimension information in a more convenient way
    def __str__(self):
        return "<class 'netcdf_viualization.ncDimension'>\n\tname: "+self.name+'\n\tsize: '+str(self.size)+'\n'

    def __repr__(self):
        return self.__str__()




class ncGriddedVar(physical_array):


    def __init__(self, name='', dataset=None, replicate_var=None,
                 grid_area=None, grid_area_units='',# grid_weight=None,
                 datatype=None, dimensions=None, shape=(), units='',
                 coordinates='', long_name='', standard_name='',
                 value=None, ancestor_value=None, link_to=None,
                 **kwargs):
        '''
        3 options to create a new ncGriddedVar
          * from a "load" object dataset
            => "dataset" kwarg is provided. Ths only information needed is "name".
          * duplicate from another ncGriddedVar
            => "replicate_var" kwarg is provided. All other information is optional, and is meant
               to modify the duplicated variable (change its name, units, ...)
          * from scratch (brand-new ncGriddedVar)
            => neither "dataset" nor "replicate_var" kwarg is provided.
               Need to provide all the information to create a ncGriddedVar from scratch (name, dimensions, ...)
        '''

        # store the potential parent information
        self.dataset = dataset


        if replicate_var is None:

            if self.dataset is None:
                # ------------------ #
                # Brand-new variable #
                # ------------------ #

                # Possibility to keep some data (source of netCDF4 dataset) from an existing ncGriddedVar
                if link_to is not None and type(link_to) != type(self):
                    raise TypeError('argument "link_to" must be a ncGriddedVar object')

                # Required attributes:
                self._nc4_dataset     = None if link_to is None else link_to._nc4_dataset
                self._nc4_parent      = None
                self._nc4_ancestor    = None if link_to is None else link_to._nc4_ancestor
                self._value           = value
                self._ancestor_value  = ancestor_value
                self.ancestor_dataset = None if link_to is None else link_to.ancestor_dataset
                self.name             = name
                self.datatype         = datatype
                self._create_dimensions_attribute(dimensions) # create 'self.dimensions' and 'self.ndim'
                self.shape            = shape
                self.units            = units

                # Optional attributes:

                if 'size' in kwargs.keys():
                    self.size = kwargs['size']

                if 'dtype' in kwargs.keys():
                    self.dtype = kwargs['dtype']
                    if self.datatype is None:
                        self.datatype = self.dtype

                else:
                    self.dtype = datatype

                if 'coordinates' in kwargs.keys():
                    self.coordinates = kwargs['coordinates']

                if 'long_name' in kwargs.keys():
                    self.long_name = kwargs['long_name']

                if 'standard_name' in kwargs.keys():
                    self.standard_name = kwargs['standard_name']

                if 'title' in kwargs.keys():
                    self.title = kwargs['title']

                # Grid attributes:

                if grid_area is None:
                    self.Grid_area = np.ones(())
                    self.Grid_area_units = ''
                else:
                    self.Grid_area = grid_area
                    self.Grid_area_units = grid_area_units

                #if grid_weight is None:
                #    self.Grid_weight = 1
                #else:
                #    self.Grid_weight = grid_weight


            else:
                # ---------------------------- #
                # Variable from load() dataset #
                # ---------------------------- #

                # Required attributes:
                self._nc4_dataset     = self.dataset._netCDF4_Dataset_original_object
                self._nc4_parent      = self._nc4_dataset.variables[name]
                self._nc4_ancestor    = self._nc4_parent
                self._value           = None
                self._ancestor_value  = None
                self.ancestor_dataset = self.dataset
                self.name             = self._nc4_parent.name
                self.datatype         = self._nc4_parent.datatype
                self._create_dimensions_attribute() # create 'self.dimensions' and 'self.ndim'
                self.shape            = self._nc4_parent.shape
                if hasattr(self._nc4_parent, 'units'):
                    self.units        = self._nc4_parent.units
                else:
                    self.units        = ''
                    print('WARNING: attribute "units" not found for variable "'+name+'". Assume unitsless variable.')

                # optional attributes:

                if hasattr(self._nc4_parent, 'size'):
                    self.size = self._nc4_parent.size

                if hasattr(self._nc4_parent, 'dtype'):
                    self.dtype = self._nc4_parent.dtype
                    if self.datatype is None:
                        self.datatype = self.dtype

                else:
                    self.dtype = self.datatype

                if hasattr(self._nc4_parent, 'coordinates'):
                    self.coordinates = self._nc4_parent.coordinates

                if hasattr(self._nc4_parent, 'long_name'):
                    self.long_name = self._nc4_parent.long_name

                if hasattr(self._nc4_parent, 'standard_name'):
                    self.standard_name = self._nc4_parent.standard_name

                if hasattr(self._nc4_parent, 'title'):
                    self.title = self._nc4_parent.title

                # Grid attributes:

                if grid_area is None:
                    self.Grid_area = np.ones(())
                    self.Grid_area_units = ''
                else:
                    self.Grid_area = grid_area
                    self.Grid_area_units = grid_area_units

                #if grid_weight is None:
                #    self.Grid_weight = 1
                #else:
                #    self.Grid_weight = grid_weight


        else:
            # ------------------------------ #
            # Duplicata of existing variable #
            # ------------------------------ #

            if type(replicate_var) != type(self):
                raise TypeError('Cannot replicate an object that is not type "ncGriddedVar"')
            else:

                # Possibility to keep the reference to the original variable
                # Should only be used if the variable info (name, units...) is unchanged!
                if link_to is not None:
                    if isinstance(link_to, bool):
                        link_to = replicate_var if link_to else None
                    elif type(link_to) != type(self):
                        raise TypeError('argument "link_to" must be a boolean or a ncGriddedVar object')

                self._nc4_dataset     = replicate_var._nc4_dataset
                self._nc4_parent      = None
                self._nc4_ancestor    = None if link_to is None else link_to._nc4_ancestor
                self._ancestor_value  = ancestor_value
                if value is None:
                    self._value       = replicate_var._value
                else:
                    self._value       = value

                # Required attributes that must be copied from variable to replicate:
                self.ancestor_dataset = replicate_var.ancestor_dataset
                self.dimensions       = replicate_var.dimensions
                self.ndim             = replicate_var.ndim
                self.shape            = replicate_var.shape

                # Required attributes that may change from variable to replicate:

                if name=='':
                    self.name         = replicate_var.name
                else:
                    self.name         = name

                if datatype is None:
                    self.datatype     = replicate_var.datatype
                else:
                    self.datatype     = datatype

                if units=='':
                    self.units        = replicate_var.units
                else:
                    self.units        = units

                # Optional attributes => 1) seek in keywords arguments, 2) seek in variable to replicate


                if 'dtype' in kwargs.keys():
                    self.dtype = kwargs['dtype']
                elif hasattr(replicate_var, 'dtype'):
                    self.dtype = replicate_var.dtype
                else:
                    self.dtype = datatype

                if self.datatype is None:
                    self.datatype = self.dtype

                if 'coordinates' in kwargs.keys():
                    self.coordinates = kwargs['coordinates']
                elif hasattr(replicate_var, 'coordinates'):
                    self.coordinates = replicate_var.coordinates

                if 'size' in kwargs.keys():
                    self.size = kwargs['size']
                elif hasattr(replicate_var, 'size'):
                    self.size = replicate_var.size

                if 'long_name' in kwargs.keys():
                    self.long_name = kwargs['long_name']
                elif hasattr(replicate_var, 'long_name'):
                    self.long_name = replicate_var.long_name

                if 'standard_name' in kwargs.keys():
                    self.standard_name = kwargs['standard_name']
                elif hasattr(replicate_var, 'standard_name'):
                    self.standard_name = replicate_var.standard_name

                if 'title' in kwargs.keys():
                    self.title = kwargs['title']
                if hasattr(replicate_var, 'title'):
                    self.title = replicate_var.title

                # Grid attributes:

                if hasattr(replicate_var, 'Grid_area') and hasattr(replicate_var, 'Grid_area_units'):
                    self.Grid_area = replicate_var.Grid_area
                    self.Grid_area_units = replicate_var.Grid_area_units
                elif grid_area is None:
                    self.Grid_area = np.ones(())
                    self.Grid_area_units = ''
                else:
                    self.Grid_area = grid_area
                    self.Grid_area_units = grid_area_units

                #if hasattr(replicate_var, 'Grid_weight'):
                #    self.Grid_weight = replicate_var.Grid_weight
                #elif grid_weight is None:
                #    self.Grid_weight = 1
                #else:
                #    self.Grid_weight = grid_weight



    #####################
    ## SPECIAL METHODS ##
    #####################


    # Method to create dimensions attribute, used in __init__
    def _create_dimensions_attribute(self, dimensions=None):

        if self._nc4_dataset is None or self._nc4_parent is None: # If orphan variable => simply use the given dimensions tuple

            if dimensions is None:
                self.dimensions = ()
            elif type(dimensions) is dict:
                self.dimensions = tuple(dim for _,dim in dimensions.items())
            else:
                self.dimensions = tuple(dimensions)

        else: # If variables derives from a netCDF4 Dataset:

            if dimensions is None:
                # If no dimension tuple given => create it (invoke 'ncDimension' class)
                self.dimensions =  tuple(ncDimension(dim, self._nc4_dataset) for dim in self._nc4_parent.dimensions)
            else:
                # Otherwise, a dictionary of 'ncDimension' object is expected (at least, of objects having the attribute 'name').
                # => Select the dimensions netCDF4 variable is defined on
                self.dimensions = tuple(dimensions[name] for name in self._nc4_parent.dimensions)

        self.ndim = len(self.dimensions)


    # Method to create the keywords arguments to pass to __init__ to create a similar variable
    # with one or several dimension(s) removed. Used by __getitem__ and by shape-reducing methods (sum, mean...).
    # WARNING: return a dictionary of kwargs (+ the list of removed dimensions), NOT a directly a ncGriddeVar object
    def _remove_dim(self, dim=None):

        if dim is None: # remove all the dimensions => create a scalar variable

            list_of_dim = []
            list_of_rm_dim = list(self.dimensions)
            grid_area = self.Grid_area.sum()

        else:

            if type(dim) is int or type(dim) is str:
                dim = [dim]

            list_of_dim = list(self.dimensions)
            list_of_rm_dim = []

            for d in dim:
                if type(d) is int:
                    if list_of_dim[d] not in list_of_rm_dim:
                        list_of_rm_dim.append(list_of_dim[d])

                    list_of_dim[d] = None
                elif type(d) is str:
                    found = False
                    for k,oldd in enumerate(self.dimensions):
                        if oldd.name == d:
                            found = True
                            if list_of_dim[k] not in list_of_rm_dim:
                                list_of_rm_dim.append(list_of_dim[k])

                            list_of_dim[k] = None

                    if not found:
                        print('Warning: dimension "'+d+'" not found')

                else:
                    print('Warning: bad dimension ID/name: ', d)

            list_of_dim = remove_all(list_of_dim, None)

            grid_area = self.Grid_area
            for d in list_of_rm_dim:
                if d._is_longitude:
                    grid_area = grid_area.sum(-1) # Longitude is last dimension (#1, or #0 is grid_area is 1D)
                elif d._is_latitude:
                    grid_area = grid_area.sum(0) # Latitude is always first dimension


        kwargs = {}

        # Mandatory attributes:
        kwargs['name']       = self.name
        kwargs['datatype']   = self.datatype
        kwargs['dimensions'] = tuple(list_of_dim)
        kwargs['shape']      = tuple(d.size for d in list_of_dim)
        kwargs['units']      = self.units

        # Optional attributes:
        if hasattr(self, 'size'):
            kwargs['size'] = np.array(kwargs['shape']).prod()

        kwargs['dtype'] = self.dtype

        if hasattr(self, 'coordinates'):
            kwargs['coordinates'] = self.coordinates

        if hasattr(self, 'long_name'):
            kwargs['long_name'] = self.long_name

        if hasattr(self, 'standard_name'):
            kwargs['standard_name'] = self.standard_name

        if hasattr(self, 'title'):
            kwargs['title'] = self.title

        # Grid attributes:
        kwargs['grid_area']       = grid_area
        kwargs['grid_area_units'] = self.Grid_area_units
        #kwargs['grid_weight']     = self.Grid_weight

        return kwargs, list_of_rm_dim


    ##################
    ## MAIN METHODS ##
    ##################


    # Redefine netCDF4 method to get the data
    def getValue(self):
        if self._nc4_parent is None:
            return self._value
        else:
            return self._nc4_parent[:]


    # Private method to generate data-compatible indices. Used for __getitem__ and __setitem__
    def _get_idx_(self, key):

        # syntax to retrieve the entire variable
        if key == slice(None):
            return self.ndim*(key,), [], self.dimensions


        if hasattr(key, '__iter__'):
            if len(key) != self.ndim:
                raise IndexError('Number of indices must match number of dimensions of ncGriddedVar (cannot reshape variable)')

        elif self.ndim == 1:
            key = (key,)

        # determine list of dimensions to remove and list of new dimensions:
        dims_to_remove = []
        new_dimensions = []
        idx_key = []
        for i,idx in enumerate(key):

            if type(idx) is int:
                dims_to_remove.append(i)
                idx_key.append(idx)
            
            elif type(idx) is slice:
                currdim = self.dimensions[i]

                if (idx.start is None or type(idx.start) is int) and (idx.stop is None or type(idx.stop) is int):
                    slc = idx
                else:
                    # Consider the slicing indices are in dimension positions (eg, latitude positions, longitude positions...)
                    # => get the actual indices of the correponding positions
                    print('Warning: slicing of dimension "'+currdim.name+'" interpreted as dimension "physical" positions')
                    #
                    orient = np.sign(currdim.positions[-1] - currdim.positions[0])
                    incpos = orient*currdim.positions
                    #
                    if idx.start is None:
                        x1 = orient*idx.stop
                        irange = np.argwhere(incpos <= x1)
                    elif idx.stop is None:
                        x0 = orient*idx.start
                        irange = np.argwhere(incpos >= x0)
                    else: # None of them are 'None'
                        x0, x1 = orient*idx.start, orient*idx.stop
                        if x0 > x1:
                            x0, x1 = x1, x0

                        irange = np.argwhere(np.logical_and((incpos >= x0), (incpos <= x1)))

                    slc = slice(irange[0][0], irange[-1][0])

                # Create new dimensions by slicing old one
                new_dimensions.append(ncDimension(currdim.name,
                                                  positions=currdim.positions[slc],
                                                  bounds=currdim.bounds[slc,:],
                                                  bounds_1D=None,
                                                  units=currdim.units,
                                                  axis=currdim.axis,
                                                  positive=currdim.positive))

                # Record current slice:
                idx_key.append(slc)

            else:
                raise IndexError('Only integers and slices (`:`) are valid indices for ncGriddedVar')

        return tuple(idx_key), dims_to_remove, tuple(new_dimensions)


    # Method for slicing (=> create a new ncGriddedVar object)
    def __getitem__(self, key):

        idx_key, dims_to_remove, new_dimensions = self._get_idx_(key)

        # Get keyword arguments to create the new GriddedVar:
        kwargs, _ = self._remove_dim(dims_to_remove)

        # Update variable-creation kwargs with new dimensions:
        kwargs['dimensions'] = new_dimensions
        kwargs['shape'] = tuple(d.size for d in new_dimensions)

        # Update Grid_area in variable-creation kwargs
        gridkey = [None, None]
        for d,k in zip(self.dimensions, idx_key):
            if d._is_latitude:
                gridkey[0] = k
            elif d._is_longitude:
                gridkey[1] = k

        gridkey = remove_all(gridkey, None)
        kwargs['grid_area'] = self.Grid_area[tuple(gridkey)]

        # Get values of variable slice
        val    = self.getValue()
        ancval = val if self._ancestor_value is None else self._ancestor_value 

        # Return new ncGriddedVar object
        return ncGriddedVar(value=val[idx_key], ancestor_value=ancval, link_to=self, **kwargs)


    # Method for modifying the data
    def __setitem__(self, key, value):

        idx_key, _, _ = self._get_idx_(key)
        if self._value is not None:
            self._value[idx_key] = value
        elif self._nc4_parent is not None:
            self._nc4_parent[idx_key] = value
        else:
            raise ReferenceError('Failed to determine where the data is stored and can be modified')


    # Redefine parent class method (used for equality test)
    def _get_data(self):
        return self.getValue()


    # Compatibility test: check if 2 objects "ncGriddedVar" are compatibles, ie: defined on the same dimensions
    def _check_compatibility(self, other, raiseerror=True):
        if self.ndim != other.ndim:
            passed = False
        else:
            passed = True
            for k in range(self.ndim):
                loc_passed = (self.dimensions[k] == other.dimensions[k])
                if not loc_passed and self.dimensions[k].name == other.dimensions[k].name and self.dimensions[k].size == 1 and other.dimensions[k].size ==1:
                    loc_passed = True
                    print('WARNING: permitted failed equality test for degenerated dimension "'+self.dimensions[k].name+'"')

                passed = (passed and loc_passed)

        if raiseerror:
            if not passed:
                raise ValueError('Variables are not defined on the same dimension(s)')

        else:
            return passed


    # Add compatibility test to parent class the equalitity test
    def __eq__(self, other):
        if super().__eq__(other):
            return self._check_compatibility(other, raiseerror=False)
        else:
            return False


    # Display variable information
    def __str__(self):
        info = ''
        info += "<class 'netcdf_viualization.ncGriddedVar'>\n"
        info += '\t'+self.name+'('
        for dim in self.dimensions[:-1]:
            info += dim.name+','

        if self.ndim > 0:
            info += self.dimensions[-1].name
        
        info += ')\t['+str(self.datatype)+']\n'

        if hasattr(self, 'long_name'):
            info += '\tname: '+self.long_name+'\n'
        elif hasattr(self, 'standard_name'):
            info += '\tname: '+self.standard_name+'\n'

        if hasattr(self, 'title'):
            info += '\ttitle: '+self.long_name+'\n'

        info += '\tunits: '+self.units+'\n'

        return info

    def __repr__(self):
        return self.__str__()



    #######################
    # Data-access methods #
    #######################


    def add_to_dataset(self, dset=None):
        '''
        Add the current ncGriddedVar to the dataset it is linked to ("ancestor_dataset"),
        to to another specified "load" dataset ("dset=...")
        '''

        if dset is None:
            if self.ancestor_dataset is None:
                raise ReferenceError('current ncGriddedVar is not linked to a "load" dataset')
            else:
                dset = self.ancestor_dataset

        elif not isinstance(dset, load):
            raise TypeError('The dataset specified in the method must be a "load" instance')

        if self.name in dset.variables:
            print('variable "'+self.name+'", already present in dataset')
        else:
            dset.create_new_var(self)


    def save_in_file(self):
        '''
        Save the ncGriddedVar data in the original netCDF file, if the ncGriddedVar is
        inherited from a netCDF file variable (for instance, through slicing).
        '''

        if self._value is None:
            if self._nc4_parent is None:
                raise ReferenceError('Could not find value to save')
            else:
                print('already saved (ncGriddedVar not detached from netCDF4 dataset)')

        elif self._nc4_ancestor is None:
            raise ReferenceError('current ncGriddedVar is not linked to a netCDF4 dataset')

        else:
            if self._ancestor_value is None:
                # Save current value in original netCDF4 file variable 
                self._nc4_ancestor[:] = self._value
            else:
                # Save the whole data field "_ancestor_value", that should have been kept up-to-date
                # with posterior modifications (like slicing) by means of array views (ie, references),
                # in original netCDF4 file variable 
                self._nc4_ancestor[:] = self._ancestor_value



    ###########################
    ## ARITHMETIC OPERATIONS ##
    ###########################


    # Define variables addition
    def __add__(self, other):

        # Mandatory conditions:
        self._check_type(other)
        self._check_compatibility(other)
        self._check_units(other)

        kwargs = {}

        if hasattr(self, 'long_name') and hasattr(other, 'long_name'):
            kwargs['long_name'] = self.long_name+' + '+other.long_name

        if hasattr(self, 'title') and hasattr(other, 'title'):
            kwargs['title'] = self.title+' + '+other.title

        if hasattr(self, 'standard_name') and hasattr(other, 'standard_name'):
            kwargs['standard_name'] = self.standard_name+'_plus_'+other.standard_name

        return ncGriddedVar(name=self.name+'_plus_'+other.name,
                            parent=None,
                            replicate_var=self, link_to=False,
                            value=self.getValue() + other.getValue(),
                            **kwargs)


    # Define variables substraction
    def __sub__(self, other):

        # Mandatory conditions:
        self._check_type(other)
        self._check_compatibility(other)
        self._check_units(other)

        kwargs = {}

        if hasattr(self, 'long_name') and hasattr(other, 'long_name'):
            kwargs['long_name'] = self.long_name+' - '+other.long_name

        if hasattr(self, 'title') and hasattr(other, 'title'):
            kwargs['title'] = self.title+' - '+other.title

        if hasattr(self, 'standard_name') and hasattr(other, 'standard_name'):
            kwargs['standard_name'] = self.standard_name+'_minus_'+other.standard_name

        return ncGriddedVar(name=self.name+'_minus_'+other.name,
                            parent=None,
                            replicate_var=self, link_to=False,
                            value=self.getValue() - other.getValue(),
                            **kwargs)


    # Define variables multiplication
    def __mul__(self, other):

        # Mandatory conditions:
        self._check_type(other)
        self._check_compatibility(other)

        kwargs = {}

        kwargs['units'] = self.units+' '+other.units

        if hasattr(self, 'long_name') and hasattr(other, 'long_name'):
            kwargs['long_name'] = self.long_name+' * '+other.long_name

        if hasattr(self, 'title') and hasattr(other, 'title'):
            kwargs['title'] = self.title+' * '+other.title

        if hasattr(self, 'standard_name') and hasattr(other, 'standard_name'):
            kwargs['standard_name'] = self.standard_name+'_times_'+other.standard_name

        return ncGriddedVar(name=self.name+'_times_'+other.name,
                            parent=None,
                            replicate_var=self, link_to=False,
                            value=self.getValue() * other.getValue(),
                            **kwargs)


    # Define variables division
    def __div__(self, other):

        # Mandatory conditions:
        self._check_type(other)
        self._check_compatibility(other)

        kwargs = {}

        kwargs['units'] = self.units+'/'+other.units

        if hasattr(self, 'long_name') and hasattr(other, 'long_name'):
            kwargs['long_name'] = self.long_name+' / '+other.long_name

        if hasattr(self, 'title') and hasattr(other, 'title'):
            kwargs['title'] = self.title+' / '+other.title

        if hasattr(self, 'standard_name') and hasattr(other, 'standard_name'):
            kwargs['standard_name'] = self.standard_name+'_over_'+other.standard_name

        return ncGriddedVar(name=self.name+'_over_'+other.name,
                            parent=None,
                            replicate_var=self, link_to=False,
                            value=self.getValue() / np.ma.masked_where(other.getValue()==0, other.getValue()),
                            **kwargs)

    def __truediv__(self, other):
        return self.__div__(other)


    # Define scalar addition
    def __radd__(self, other):
        # TO DO IN THE FUTURE: test if "other" has a units

        kwargs = {}

        if hasattr(self, 'long_name'):
            kwargs['long_name'] = str(other)+' + '+self.long_name

        if hasattr(self, 'title'):
            kwargs['title'] = str(other)+' + '+self.title

        if hasattr(self, 'standard_name'):
            kwargs['standard_name'] = str(other)+'_plus_'+self.standard_name

        return ncGriddedVar(name=self.name,
                            parent=None,
                            replicate_var=self, link_to=False,
                            value=other+self.getValue(),
                            **kwargs)


    # Define scalar substraction
    def __rsub__(self, other):
        # TO DO IN THE FUTURE: test if "other" has a units

        kwargs = {}

        if hasattr(self, 'long_name'):
            kwargs['long_name'] = str(other)+' - '+self.long_name

        if hasattr(self, 'title'):
            kwargs['title'] = str(other)+' - '+self.title

        if hasattr(self, 'standard_name'):
            kwargs['standard_name'] = str(other)+'_minus_'+self.standard_name

        return ncGriddedVar(name=self.name,
                            parent=None,
                            replicate_var=self, link_to=False,
                            value=other-self.getValue(),
                            **kwargs)


    # Define scalar multpiplication
    def __rmul__(self, other):
        # TO DO IN THE FUTURE: test if "other" has a units a update self.units 

        kwargs = {}

        if hasattr(self, 'long_name'):
            kwargs['long_name'] = str(other)+' * '+self.long_name

        if hasattr(self, 'title'):
            kwargs['title'] = str(other)+' * '+self.title

        if hasattr(self, 'standard_name'):
            kwargs['standard_name'] = str(other)+'_times_'+self.standard_name

        return ncGriddedVar(name=self.name,
                            parent=None,
                            replicate_var=self, link_to=False,
                            value=other*self.getValue(),
                            **kwargs)


    # Define variables or scalar power operator
    def __pow__(self, other):

        if self._check_type(other, raiseerror=False):

            # Mandatory conditions:
            self._check_compatibility(other)

            kwargs = {}

            kwargs['units'] = self.units+'/'+other.units

            if hasattr(self, 'long_name') and hasattr(other, 'long_name'):
                kwargs['long_name'] = self.long_name+' ^ ('+other.long_name+')'

            if hasattr(self, 'title') and hasattr(other, 'title'):
                kwargs['title'] = self.title+' ^ ('+other.title+')'

            if hasattr(self, 'standard_name') and hasattr(other, 'standard_name'):
                kwargs['standard_name'] = self.standard_name+'_power_'+other.standard_name

            return ncGriddedVar(name=self.name+'_power_'+other.name,
                                parent=None,
                                replicate_var=self, link_to=False,
                                value=self.getValue() ** other.getValue(),
                                **kwargs)

        else:

            kwargs = {}

            if hasattr(self, 'long_name'):
                kwargs['long_name'] = self.long_name+' ^ '+str(other)

            if hasattr(self, 'title'):
                kwargs['title'] = self.title+' ^ '+str(other)

            if hasattr(self, 'standard_name'):
                kwargs['standard_name'] = self.standard_name+'_power_'+str(other)

            return ncGriddedVar(name=self.name,
                                parent=None,
                                replicate_var=self, link_to=False,
                                value=self.getValue()**other,
                                **kwargs)


    # Method to create the n-dimensional volume of "grid" the variable is defined on
    # Takes into account horizontal grid area if the variable is defined on longitude and/or latitude
    def _get_grid_volume(self):

        volume = np.ones(self.shape)

        # Volume for all dimensions expect potential longitude and latitude:
        remain_dim = []
        for k,d in enumerate(self.dimensions):
            if d._is_longitude:
                remain_dim.append(d)
            elif d._is_latitude:
                remain_dim.append(d)
            else:
                shp = self.ndim * [1]
                slc = self.ndim * [0]
                shp[k] = d.size
                slc[k] = slice(d.size)
                dvol = np.ones(tuple(shp))
                dvol[tuple(slc)] = d.bounds[:,1]-d.bounds[:,0]
                volume = volume * dvol

        if len(remain_dim) == 2: # longitude and latitude:

            grid_area = self.Grid_area
            if remain_dim[0]._is_longitude: # longitude first => reverse axis
                grid_area = grid_area.transpose()

            i = self.dimensions.index(remain_dim[0])
            j = self.dimensions.index(remain_dim[1])
            shp = self.ndim * [1]
            slc = self.ndim * [0]
            shp[i],shp[j] = remain_dim[0].size, remain_dim[1].size
            slc[i],slc[j] = slice(remain_dim[0].size), slice(remain_dim[1].size)
            dvol = np.ones(tuple(shp))
            dvol[tuple(slc)] = grid_area
            volume = volume * dvol

        elif len(remain_dim) == 1:

            i = self.dimensions.index(remain_dim[0])
            shp = self.ndim * [1]
            slc = self.ndim * [0]
            shp[i] = remain_dim[0].size
            slc[i] = slice(remain_dim[0].size)
            dvol = np.ones(tuple(shp))

            if remain_dim[0]._is_longitude:
                dvol[tuple(slc)] = self.Grid_area.mean(0) # Average grid area on latitude (1st dim of Grid_area)
            elif remain_dim[0]._is_latitude:
                dvol[tuple(slc)] = self.Grid_area.mean(1) # Average grid area on longitude (2nd dim of Grid_area)

            volume = volume * dvol

        return volume
           


    # Average variable along one or several dimensions:
    def mean(self, dim=None):

        kwargs, mean_dim = self._remove_dim(dim)
        mean_dim_index = tuple(self.dimensions.index(d) for d in mean_dim)

        value = self.getValue()
        volume = self._get_grid_volume()
        volume[value.mask] = 0

        return ncGriddedVar(value=(value*volume).sum(mean_dim_index) / volume.sum(mean_dim_index),
                            **kwargs)


    # Sum variable along one or several dimensions:
    def sum(self, dim=None):

        kwargs, sum_dim = self._remove_dim(dim)
        sum_dim_index = tuple(self.dimensions.index(d) for d in sum_dim)

        return ncGriddedVar(value=self.getValue().sum(sum_dim_index), **kwargs)


    # Integrate variable along one or several dimensions:
    def integrate(self, dim=None):

        kwargs, integ_dim = self._remove_dim(dim)
        integ_dim_index = tuple(self.dimensions.index(d) for d in integ_dim)

        # Multiply units by dimension units:
        lon,lat = False,False
        for d in integ_dim:
            if d._is_longitude:
                lon = True
            elif d._is_latitude:
                lat = True
            else:
                kwargs['units'] = kwargs['units'] + ' ' + d.units

        if lon and lat:
            kwargs['units'] = kwargs['units'] + ' ' + self.Grid_area_units
        elif lon or lat:
            kwargs['units'] = kwargs['units'] + ' m'

        value = self.getValue()
        volume = self._get_grid_volume()

        return ncGriddedVar(value=(value*volume).sum(integ_dim_index), **kwargs)


    # Maximum value along one or several dimensions:
    def max(self, dim=None):
        kwargs, max_dim = self._remove_dim(dim)
        max_dim_index = tuple(self.dimensions.index(d) for d in max_dim)
        value = self.getValue()
        return ncGriddedVar(value=value.max(max_dim_index), **kwargs)


    # Minimum value along one or several dimensions:
    def min(self, dim=None):
        kwargs, min_dim = self._remove_dim(dim)
        min_dim_index = tuple(self.dimensions.index(d) for d in min_dim)
        value = self.getValue()
        return ncGriddedVar(value=value.min(min_dim_index), **kwargs)



    ######################
    ## PLOTTING METHODS ##
    ######################

    # Method to get list of dimensions which length is greater than 1, and dimension order (X,Y,Z,T,...)
    def _get_dimensions_order(self):
        dim_num   = []
        dim_order = []
        shape     = []
        for k,dim in enumerate(self.dimensions):
            if (dim.size != 1):
                dim_num.append(k)
                shape.append(dim.size)
                if dim.axis.upper() == 'X':
                    dim_order.append(1)
                elif dim.axis.upper() == 'Y':
                    dim_order.append(2)
                elif dim.axis.upper() == 'Z':
                    dim_order.append(3)
                elif dim.axis.upper() == 'T':
                    dim_order.append(0)
                else:
                    dim_order.append(1.5)

        return dim_num,dim_order,tuple(shape)


    # Put variable title on figure
    def _get_title(self):
        for attribute in ['title', 'long_name', 'standard_name', 'name']:
            if hasattr(self, attribute):
                return getattr(self, attribute)

    def put_title(self, ax=None, has_colorbar=False, override_title=None):

        if ax is None:
            ax = plt.gca()

        if override_title is None:
            text = self._get_title()
        else:
            text = override_title

        if not has_colorbar:
            text += ' ('+self.units+')'

        ax.set_title(text)



    # 1D plot
    # =======

    def plot(self, fig=None, ax=None, overlay=False, o=None, title=True, show=True, s=None, **plot_kwargs):

        # Key-word shortcuts:
        #
        if o is not None:
            overlay = o
        #
        if s is not None:
            show = s

        dimid,dimorder,shp = self._get_dimensions_order()

        if len(shp) != 1:
            print('Cannot plot a variable that is not 1-dimensionnal')
        else:
            fig, ax = get_axis_and_figure(fig, ax, overlay)

            geoaxis = []

            if dimorder[0] < 2:
                # + + + + + + + + + + + + + + + + + + + + + #
                ax.plot(self.dimensions[dimid[0]].positions,
                        self.getValue().reshape(shp),
                        **plot_kwargs)
                # + + + + + + + + + + + + + + + + + + + + + #

                if self.dimensions[dimid[0]]._is_longitude:
                    geoaxis.append('x')

                if self.dimensions[dimid[0]].positive == 'down':
                    ax.invert_xaxis()

            else:
                # + + + + + + + + + + + + + + + + + + + + + #
                ax.plot(self.getValue().reshape(shp),
                        self.dimensions[dimid[0]].positions,
                        **plot_kwargs)
                # + + + + + + + + + + + + + + + + + + + + + #

                if self.dimensions[dimid[0]]._is_latitude:
                    geoaxis.append('y')

                if self.dimensions[dimid[0]].positive == 'down':
                    ax.invert_yaxis()


            # Ticks
            for geoax in geoaxis: 
                geographic_ticks(ax, axis=geoax)

            # Title
            if type(title) is str:
                new_title = title
                title = True
            else:
                new_title = None

            if title:
                self.put_title(ax=ax, has_colorbar=False, override_title=new_title)

            # Draw plot
            if show:
                plt.show()


    # 2D generic plot
    # ===============

    def _plot2D(self, which, fig, ax, overlay, cax, cbar, clim, title, show, **plot2D_kwargs):

        dimid,dimorder,shp = self._get_dimensions_order()

        if len(shp) != 2:
            print('Cannot '+which+' a variable that is not 2-dimensionnal')
        else:
            fig, ax = get_axis_and_figure(fig, ax, overlay)

            data = self.getValue()

            # clim argument:
            if clim is not None:

                if clim == 'c' or clim == 'centrered': #=> colormap range centered on 0

                    if 'vmin' not in plot2D_kwargs and 'vmax' not in plot2D_kwargs:
                        # + + + + + + + + + + + + + #
                        absmax = (np.abs(data)).max()
                        # + + + + + + + + + + + + + #
                        plot2D_kwargs['vmin'] = -absmax
                        plot2D_kwargs['vmax'] = absmax
                    else:
                        if 'vmin' not in plot2D_kwargs:
                            plot2D_kwargs['vmin'] = -plot2D_kwargs['vmax']
                        elif 'vmax' not in plot2D_kwargs:
                            plot2D_kwargs['vmax'] = -plot2D_kwargs['vmin']
                        else:
                            absmax = max(-plot2D_kwargs['vmin'], plot2D_kwargs['vmax'])
                            plot2D_kwargs['vmin'] = -absmax
                            plot2D_kwargs['vmax'] = absmax

                else:

                    try:
                        plot2D_kwargs['vmin'] = clim[0]
                        plot2D_kwargs['vmax'] = clim[1]
                    except:
                        print('Ignored invalid argument "clim".')


            # shade/fill/contour specific variables
            if which == 'shade':

                x = self.dimensions[dimid[0]].bounds_1D
                y = self.dimensions[dimid[1]].bounds_1D
                plot_command = ax.pcolormesh

            else:
            
                x = self.dimensions[dimid[0]].positions
                y = self.dimensions[dimid[1]].positions
                if which == 'fill':
                    plot_command = ax.contourf
                elif which == 'contour':
                    plot_command = ax.contour

                # Specific case: for contour and contourf, "vmin" and "vmax" arguments do not work:
                if 'vmin' in plot2D_kwargs and 'vmax' in plot2D_kwargs:
                    if 'levels' in plot2D_kwargs:
                        try:
                            plot2D_kwargs['levels'] = np.linspace(plot2D_kwargs['vmin'],
                                                                  plot2D_kwargs['vmax'],
                                                                  plot2D_kwargs['levels']+1)
                            new = False
                        except:
                            print('Override "levels" argument')
                            new = True
                    else:
                        new = True

                    if new:
                        plot2D_kwargs['levels'] = np.linspace(plot2D_kwargs['vmin'],
                                                              plot2D_kwargs['vmax'],
                                                              16)


            if dimorder[1]<=dimorder[0]:
                # + + + + + + + + + + + + + + + + + + + + + + + + + + + + + #
                pid = plot_command(y, x, data.reshape(shp), **plot2D_kwargs)
                # + + + + + + + + + + + + + + + + + + + + + + + + + + + + + #
                ix, iy = 1, 0
            else:
                # + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + #
                pid = plot_command(x, y, data.reshape(shp).transpose(),**plot2D_kwargs)
                # + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + #
                ix, iy = 0, 1


            # Customize plot:

            geoaxis = []

            if self.dimensions[dimid[ix]]._is_longitude:
                geoaxis.append('x')

            if self.dimensions[dimid[iy]]._is_latitude:
                geoaxis.append('y')

            if self.dimensions[dimid[ix]].positive == 'down':
                ax.invert_xaxis()

            if self.dimensions[dimid[iy]].positive == 'down':
                ax.invert_yaxis()

            # Ticks
            for geoax in geoaxis:
                geographic_ticks(ax, axis=geoax)

            # Colorbar
            if cbar:
                cid = fig.colorbar(pid, cax=cax)

                # colorbar legend
                if cid.orientation == 'vertical':
                    cid.ax.set_ylabel(self.units)
                elif cid.orientation == 'horizontal':
                    cid.ax.set_xlabel(self.units)

            # Title
            if type(title) is str:
                new_title = title
                title = True
            else:
                new_title = None

            if title:
                self.put_title(ax=ax, has_colorbar=cbar, override_title=new_title)

            # Draw plot
            if show:
                plt.show()


    # 2D raw plot
    # ===========

    def shade(self,
              fig=None, ax=None, overlay=False, o=None, cax=None, cbar=True, clim=None, title=True, show=True, s=None,
              **shade_kwargs):

        # Key-word shortcuts:
        #
        if o is not None:
            overlay = o
        #
        if s is not None:
            show = s

        self._plot2D(which='shade',
                     fig=fig, ax=ax, overlay=overlay, cax=cax, cbar=cbar, clim=clim, title=title, show=show,
                     **shade_kwargs)


    # 2D smoothed plot
    # ================

    def fill(self,
             fig=None, ax=None, overlay=False, o=None, cax=None, cbar=True, clim=None, title=True, show=True, s=None,
             **fill_kwargs):

        # Key-word shortcuts:
        #
        if o is not None:
            overlay = o
        #
        if s is not None:
            show = s

        self._plot2D(which='fill',
                     fig=fig, ax=ax, overlay=overlay, cax=cax, cbar=cbar, clim=clim, title=title, show=show,
                     **fill_kwargs)


    # 2D contour plot
    # ===============

    def contour(self,
                fig=None, ax=None, overlay=False, o=None, cax=None, cbar=False, clim=None, title=True, show=True, s=None,
                **contour_kwargs):

        # Key-word shortcuts:
        #
        if o is not None:
            overlay = o
        #
        if s is not None:
            show = s

        self._plot2D(which='contour',
                     fig=fig, ax=ax, overlay=overlay, cax=cax, cbar=cbar, clim=clim, title=title, show=show,
                     **contour_kwargs)


    # ============================================ #
    # Method to create an interactive editing plot #
    # ============================================ #

    def shaedit(self, fig=None, ax=None, cax=None, cmap=None, clim=None, show=True):
        # Keep only non-degenerated dimensions
        dimid,_,_ = self._get_dimensions_order()
        tslc = tuple(slice(None) if k in dimid else 0 for k in range(self.ndim))

        self._dummy_ = shaedit(self[tslc], fig=fig, ax=ax, cax=cax, cmap=cmap, clim=clim, show=show)




class load():


    def __init__(self, filename, mode='r', **kwargs):

        if mode in ('r', 'r+'):
            pass
        elif mode == 'w':
            raise ValueError('Illegal mode=="r". Cannot create new dataset with "load"')
        else:
            raise ValueError('Illegal value "{:}" for "mode"'.format(mode))

        # Invoke "netCDF4 Dataset" (and store the object as a private attribute):
        self._netCDF4_Dataset_original_object = netCDF4.Dataset(filename, mode=mode, **kwargs)

        # Copy the main attributes of "nc.Dataset" object:
        self.mode = mode

        # Always present attributes:
        self.file_format  = self._netCDF4_Dataset_original_object.file_format
        self.disk_format  = self._netCDF4_Dataset_original_object.disk_format
        self.filepath     = self._netCDF4_Dataset_original_object.filepath()
        self.path         = self._netCDF4_Dataset_original_object.path
        self.parent       = self._netCDF4_Dataset_original_object.parent
        self.groups       = self._netCDF4_Dataset_original_object.groups
        self.cmptypes     = self._netCDF4_Dataset_original_object.cmptypes
        self.data_model   = self._netCDF4_Dataset_original_object.data_model
        self.enumtypes    = self._netCDF4_Dataset_original_object.enumtypes
        self.vltypes      = self._netCDF4_Dataset_original_object.vltypes
        # Optional attributes
        if hasattr(self._netCDF4_Dataset_original_object, 'FileHeader'):
            self.FileHeader = self._netCDF4_Dataset_original_object.FileHeader

        if hasattr(self._netCDF4_Dataset_original_object, 'FileInfo'):
            self.FileInfo   = self._netCDF4_Dataset_original_object.FileInfo

        if hasattr(self._netCDF4_Dataset_original_object, 'GridHeader'):
            self.GridHeader = self._netCDF4_Dataset_original_object.GridHeader

        if hasattr(self._netCDF4_Dataset_original_object, 'history'):
            self.history    = self._netCDF4_Dataset_original_object.history


        # Dimensions:
        # -----------

        self.dimensions = {}
        for key in self._netCDF4_Dataset_original_object.dimensions.keys():
            ##############################################################################
            self.dimensions[key] = ncDimension(key, self._netCDF4_Dataset_original_object)
            ##############################################################################


        # Check longitude-latitude

        self._longitude = None
        self._latitude  = None
        self._geographic_grid = True

        for key in self.dimensions.keys():

            if self.dimensions[key]._is_longitude:
                if self._longitude is None:
                    self._longitude = self.dimensions[key]
                    self._longitude.axis = 'X'
                else:
                    print('WARNING: more than 1 dimension identified as longitude.')
                    print('Keep the first found: "'+self._longitude.name+'" and ignore "'+self.dimensions[key].name+'"')
                    self.dimensions[key]._is_longitude = False

            if self.dimensions[key]._is_latitude:
                if self._latitude is None:
                    self._latitude = self.dimensions[key]
                    self._latitude.axis = 'Y'
                else:
                    print('WARNING: more than 1 dimension identified as latitude.')
                    print('Keep the first found: "'+self._latitude.name+'" and ignore "'+self.dimensions[key].name+'"')
                    self.dimensions[key]._is_latitude = False

        if self._longitude is None or self._latitude is None:
            print('WARNING: cannot identify longitude or latitude, assume grid is not geographic')
            self._geographic_grid = False
        else:
            if self._longitude.units not in LONGITUDE_LEGAL_UNITS:
                print('WARNING: longitude units "'+self._longitude.units+'" not recognized. Use of geographic grid failed')
                self._longitude._is_longitude = False
                self._geographic_grid = False
            else:
                self._longitude._check_bounds()

            if self._latitude.units not in LATITUDE_LEGAL_UNITS:
                print('WARNING: latitude units "'+self._latitude.units+'" not recognized. Use of geographic grid failed')
                self._latitude._is_latitude = False
                self._geographic_grid = False
            else:
                self._latitude._check_bounds()


        # Earth model and horizontal weighting:
        # -------------------------------------

        if 'area' in self._netCDF4_Dataset_original_object.variables.keys():
            areavar = 'area'
        if 'cell_area' in self._netCDF4_Dataset_original_object.variables.keys():
            areavar = 'cell_area'
        if 'grid_area' in self._netCDF4_Dataset_original_object.variables.keys():
            areavar = 'grid_area'
        if 'grid_cell_area' in self._netCDF4_Dataset_original_object.variables.keys():
            areavar = 'grid_cell_area'
        else:
            areavar = None

        if areavar is not None:
            if self._netCDF4_Dataset_original_object.variables[areavar].ndim == 2:
                self.Grid_area = self._netCDF4_Dataset_original_object.variables[areavar][:,:].data
                self.Grid_area[self._netCDF4_Dataset_original_object.variables[areavar][:,:].mask] = 0
                self.Grid_area_units = self._netCDF4_Dataset_original_object.variables[areavar].units

        else:
            self.Grid_area = np.array(1.)
            self.Grid_area_units = ''


        self.Earth_model = {}

        if self._geographic_grid:

            self.Earth_model['datum'] = 'not yet defined'
            if hasattr(self._netCDF4_Dataset_original_object, 'earth_model'):
                self.Earth_model['datum'] = self._netCDF4_Dataset_original_object.earth_model.lower()
            elif hasattr(self._netCDF4_Dataset_original_object, 'Earth_model'):
                self.Earth_model['datum'] = self._netCDF4_Dataset_original_object.Earth_model.lower()

            if self.Earth_model['datum'] == 'spherical':
                self.Earth_model['semi_major_axis'] = 6371007.2 # authalic radius (m)
                self.Earth_model['semi_minor_axis'] = 6371007.2
                self.Earth_model['flattening']      = 0.
                self.Earth_model['eccentricity']    = 0.
            else:
                if self.Earth_model != 'WGS84':
                    print('WARNING: unkown Earth model. Assumed WGS84.')

                self.Earth_model['datum']           = 'WGS84'
                self.Earth_model['semi_major_axis'] = 6378137.0 #(m)
                self.Earth_model['semi_minor_axis'] = 6356752.314245 #(m)
                self.Earth_model['flattening']      = 0.0033528106647474805
                self.Earth_model['eccentricity']    = 0.08181919084262157

            if self.Grid_area == np.array(1.):
                print('WARNING: horizontal grid area automatically computed using Earth model')
                self.Grid_area = cell_area( self._longitude.bounds[:,0], self._latitude.bounds[:,0],
                                            self._longitude.bounds[:,1], self._latitude.bounds[:,1],
                                            self.Earth_model['semi_major_axis'], self.Earth_model['eccentricity'])
                self.Grid_area = self.Grid_area.transpose()
                self.Grid_area_units = 'm2'


        #if 'contfrac' in self._netCDF4_Dataset_original_object.variables.keys():
        #    fracvar = 'contfrac'
        #if 'cont_frac' in self._netCDF4_Dataset_original_object.variables.keys():
        #    fracvar = 'cont_frac'
        #if 'landfrac' in self._netCDF4_Dataset_original_object.variables.keys():
        #    fracvar = 'land_frac'
        #if 'land_frac' in self._netCDF4_Dataset_original_object.variables.keys():
        #    fracvar = 'land_frac'
        #else:
        #    fracvar = None
        #
        #if fracvar is not None:
        #    if self._netCDF4_Dataset_original_object.variables[fracvar].shape == self.Grid_area.shape:
        #        print('NOTE: variable "'+fracvar+'" used as "valid" fraction of grid cell for weighting')
        #        self.Grid_weight = self.Grid_area * self._netCDF4_Dataset_original_object.variables[fracvar][:,:].data
        #        self.Grid_weight[self._netCDF4_Dataset_original_object.variables[fracvar][:,:].mask] = 0
        #        self.Grid_weight = self.Grid_weight / np.sum(self.Grid_weight)
        #
        #else:
        #    self.Grid_weight = self.Grid_area / np.sum(self.Grid_area)


        # ----------
        # Variables:
        # ----------

        self.variables = {}
        for key in self._netCDF4_Dataset_original_object.variables.keys():
            ##########################################################################################################
            self.variables[key] = ncGriddedVar(key, dataset=self,
                                               grid_area=self.Grid_area, grid_area_units=self.Grid_area_units)#, 
                                               #grid_weight=self.Grid_weight)
            ##########################################################################################################



    # Redefine __getitem__ method as a shortcut to variables dictionary
    def __getitem__(self, key):
        return self.variables.__getitem__(key)


    # Display data information in a more convenient way
    def __str__(self):
        info  =  '\n'
        info  +=  "<class 'netcdf_viualization.load'> (netCDF dataset)\n\n"
        info  +=  "from: '"+self.filepath+"'\n"
        info  +=  self.file_format+' data model, file format '+self.disk_format+'\n\n'
        if hasattr(self, 'title'):
            info += 'title: '+self.long_name+'\n\n'

        info  +=  'dimensions(sizes):\n'
        for dim in self.dimensions:
            info += '\t'+dim+'('+str(self.dimensions[dim].size)+')\n'

        info  +=  '\n'
        info  +=  'variables(dimensions):\n'
        for var in self.variables:
            info += '\t'+var+'('
            for dim in self[var].dimensions[:-1]:
                info += dim.name+','

            if len(self[var].dimensions) > 0:
                info += self[var].dimensions[-1].name

            info += ')\t['+str(self[var].datatype)+']\n'

        if self._geographic_grid:
            info  +=  '\n'
            info  +=  'Earth model: '+self.Earth_model['datum']

        info  +=  '\n'

        return info

    def __repr__(self):
        return self.__str__()


    def create_new_var(self, var: ncGriddedVar):
        '''
        Method to add a new variable to the current dataset open with "load".
        "var" must be an ncGriddedVar object.
        This method only works in "r+" mode (read-write access)
        '''

        if not isinstance(var, ncGriddedVar):
            raise TypeError('"var" argument must be a ncGriddedVar instance')

        if self.mode == 'r':
            raise PermissionError('Cannot add new variable in read-only mode.')

        list_dim_name = []
        for dim in var.dimensions:
            if not isinstance(dim, ncDimension):
                raise TypeError('All dimension of ncGriddedVar "var" must be ncDimension instances')
            elif dim not in self.dimensions.values():
                raise ValueError('Dimension "'+dim.name+'" of ncGridded is not in current dataset')
            else:
                list_dim_name.append(dim.name)

        # Put variable in netCDF4 dataset
        # -------------------------------

        # Fill-value
        fill_value=None
        for fval in ['missing_value', 'fill_value', '_FillValue']:
            if hasattr(var, fval):
                fill_value = getattr(var, fval)
                break
            elif hasattr(var._value, fval):
                fill_value = getattr(var._value, fval)
                break

        # Variable definition
        vid = self._netCDF4_Dataset_original_object.createVariable(var.name, datatype=var.datatype,
                                                                   dimensions=list_dim_name, fill_value=fill_value)

        # Copy variable attributes
        for att in ['name', 'units', 'standard_name', 'long_name']:
            if hasattr(var, att):
                vid.setncattr(att, getattr(var, att))

        # Copy variable data
        vid[:] = var.getValue()


        # Update current "load" dataset information
        # -----------------------------------------
        self.variables[var.name] = var

        # Link variable "var" to current dataset
        # --------------------------------------
        var.dataset          = self
        var.ancestor_dataset = self
        var._value           = None
        var._ancestor_value  = None
        var._nc4_dataset     = self._netCDF4_Dataset_original_object
        var._nc4_parent      = vid
        var._nc4_ancestor    = vid



    # =========================================================================== #
    # Method to create an interactive editing plot of all the variables contained #
    # =========================================================================== #

    def shaedit(self, exclude=(), fig=None, ax=None, cax=None, cmap=None, clim=None, show=True):
        self._dummy_ = shaedit(*(var for key,var in self.variables.items() if key not in self.dimensions and key not in exclude),
                               fig=fig, ax=ax, cax=cax, cmap=cmap, clim=clim, show=show)



    # Method to close netCDF parent file
    def close(self):
        self._netCDF4_Dataset_original_object.close()




#####################################################
##                                                 ##
##  Class to interactively edit gridded variables  ##
##                                                 ##
#####################################################



class shaedit():
    '''
    Create an interactive plot to edit 2D variables (*args).
    Input arguments (*args) are expected to be numpy.ndarray, netCDF4.Variable or ncGriddedVar instances.
    If using numpy-arrays, to be able to retrieve the arrays after editing them, do:
    `obj = shaedit(*args)`
    The arrays will be stored in `obj.arrays`
    '''

    def __init__(self, *args, fig=None, ax=None, cax=None, cmap=None, clim=None, show=True):
        '''
        *args must be numpy.ndarray, netCDF4.Variable or ncGriddedVar instances.
        Cf docstring of "shaedit" class.
        '''

        if len(args) == 0:
            raise ValueError('At least 1 argument expected to create an interactive editing plot')

        for var in args:
            if not isinstance(var, np.ndarray) and \
               not isinstance(var, ncGriddedVar) and \
               not isinstance(var, netCDF4.Variable):
                raise TypeError('Input arguments must be numpy arrays, netCDF4.Variable or ncGriddedVar instances')

        shape = None
        variables = []
        for var in args:
            if len(var.shape) == 2:
                if shape is None:
                    shape = var.shape

                if var.shape != shape:
                    print('Warning: skipped 2D variable whose shape is inconcsistent with previous variable(s)')
                else:
                    variables.append(var)

            else:
                print('Warning: skipped non 2-dimensional variable')

        if shape is None: # => no variable were loaded
            raise ValueError('At least one 2-dimensional variable is expected')

        # Record main elements
        self.variables = variables
        self.arrays = [np.ma.masked_where(False, (var.getValue() if isinstance(var, ncGriddedVar) else var), copy=True)
                       for var in variables]
        self._ivar = 0
        self.current_var = self.variables[self._ivar]
        self.current_array = self.arrays[self._ivar]
        self._data_cache = None
        self.shape = shape

        # Data dimension
        self.yi, self.xi = (np.arange(0, size+1, 1) for size in self.shape)
        self.nav_xi, self.nav_yi = np.meshgrid((self.xi[:-1]+self.xi[1:])/2, (self.yi[:-1]+self.yi[1:])/2)

        # Editing mask
        mask = np.zeros(self.shape, dtype='int8')
        self.editmask = np.ma.masked_where(~mask.astype(bool), mask, copy=False)

        # Create background figure
        # ------------------------
        if ax is None:
            if fig is None:
                fig = plt.figure(figsize=(15,10))

            ax = fig.add_axes([.20, .13, .86, .70])

        self.fig = fig
        self.ax  = ax

        # Plot current variable
        # ---------------------
        vmin, vmax = (None,None) if clim is None else clim
        self.pid = ax.pcolormesh(self.xi, self.yi, self.current_array,
                                 cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        self.mpid = ax.pcolormesh(self.xi, self.yi, self.editmask, cmap='spring', vmin=0.8, vmax=1.5, alpha=0.5, zorder=2)
        self.cbid = fig.colorbar(self.pid, ax=ax, cax=cax)
        if hasattr(self.current_var, 'name'):
            self.ax.set_title(self.current_var.name)


        # Replicate plot information for all variables in the list
        self.list_clim = [self.pid.get_clim()] + [(x.min(), x.max()) for x in self.arrays[1:]]
        self.list_cmap = len(self.arrays)*[self.pid.get_cmap()]
        self.current_clim = self.list_clim[self._ivar]
        self.current_cmap = self.list_cmap[self._ivar]

        

        # ======================= #
        # Create data-editing GUI #
        # ======================= #

        self.gui = {}


        # Selectors
        # ---------

        self.selecting = True

        self.gui['RecSel'] = mp_wdgt.RectangleSelector(self.ax, onselect=self._RecSel_Updte_, interactive=False)
        self.gui['RecSel'].set_active(True)

        self.gui['PolSel'] = mp_wdgt.PolygonSelector(self.ax, onselect=self._PolSel_Updte_)
        self.gui['PolSel'].set_active(False)

        self.gui['LasSel'] = mp_wdgt.LassoSelector(self.ax, onselect=self._PolSel_Updte_)
        self.gui['LasSel'].set_active(False)


        # Buttons activating selectors
        # ----------------------------

        self.gui['selector'] = mp_wdgt.RadioButtons(self.fig.add_axes([0.02, 0.85, 0.12, 0.13]),
                                                      ['rectangle selector', 'polygon selector', 'lasso selector'], active=0)
        self.gui['selector'].on_clicked(self._Swtch_Selector_)

        self.gui['select/unselect'] = mp_wdgt.RadioButtons(self.fig.add_axes([0.02, 0.74, 0.08, 0.07]),
                                                           ['select', 'unselect'], active=0)
        self.gui['select/unselect'].on_clicked(self._Swtch_SelUnsel_)

        self.gui['clear select'] = mp_wdgt.Button(self.fig.add_axes([0.02, 0.65, 0.12, 0.05]),
                                                  'clear selection')
        self.gui['clear select'].on_clicked(self._Clear_EditMask_)


        # Buttons editing data within selected area
        # -----------------------------------------

        self.data_edit_props = {'where': 'unmasked data', 'how': 'absolute'}

        self.gui['edit'] = mp_wdgt.Button(self.fig.add_axes([0.45, 0.90, 0.08, 0.08]),
                                          'replace\nselection')
        self.gui['edit'].on_clicked(self._Edit_Sel_)

        self.gui['edit where'] = mp_wdgt.RadioButtons(self.fig.add_axes([0.55, 0.90, 0.13, 0.08]),
                                                      ['unmasked data', 'masked data', 'both'], active=0)
        self.gui['edit where'].on_clicked(self._Set_EditWhere_)
        self.gui['edit how']   = mp_wdgt.RadioButtons(self.fig.add_axes([0.70, 0.90, 0.10, 0.08]),
                                                      ['absolute', 'delta', '%'], active=0)
        self.gui['edit how'].on_clicked(self._Set_EditHow_)

        self.gui['value'] = mp_wdgt.TextBox(self.fig.add_axes([0.83, 0.92, 0.07, 0.04]),
                                            '', textalignment='center')
        self.gui['value'].on_submit(self._Set_Edit_Value_)

        self.gui['confirm'] = mp_wdgt.Button(self.fig.add_axes([0.93, 0.945, 0.05, 0.04]),
                                             'confirm', color='lightsalmon', hovercolor='mistyrose')
        self.gui['confirm'].on_clicked(self._Confirm_Edit_)
        self.gui['cancel']  = mp_wdgt.Button(self.fig.add_axes([0.93, 0.895, 0.05, 0.04]),
                                             'cancel', color='lightsalmon', hovercolor='mistyrose')
        self.gui['cancel'].on_clicked(self._Cancel_Edit_)

        for key in ('value', 'confirm', 'cancel'):
            self.gui[key].ax.set_visible(False)


        # Buttons handling the list of variables to display (and edit)
        # ------------------------------------------------------------

        self.gui['varnum'] = self.fig.text(0.285, 0.06, '1/{:}'.format(len(self.arrays)),
                                           ha='center', fontsize=15)
        self.gui['next'] = mp_wdgt.Button(self.fig.add_axes([0.29, 0.01, 0.08, 0.03]),
                                          'NEXT VAR')
        self.gui['next'].on_clicked(self._Iter_NextVar_)
        self.gui['prev'] = mp_wdgt.Button(self.fig.add_axes([0.20, 0.01, 0.08, 0.03]),
                                          'PREV VAR')
        self.gui['prev'].on_clicked(self._Iter_PrevVar_)
        self.gui['new var'] = mp_wdgt.Button(self.fig.add_axes([0.07, 0.06, 0.11, 0.04]),
                                             'CREATE NEW VAR')
        self.gui['new var'].on_clicked(self._Add_NewVar_)
        self.gui['duplicate'] = mp_wdgt.Button(self.fig.add_axes([0.07, 0.01, 0.11, 0.04]),
                                               'DUPLICATE VAR')
        self.gui['duplicate'].on_clicked(self._Add_CopyVar_)

        self.gui['write'] = mp_wdgt.Button(self.fig.add_axes([0.50, 0.025, 0.15, 0.07]),
                                           'SAVE', color='pink', hovercolor='mistyrose')
        self.gui['write'].on_clicked(self._write_to_file_)



        # Buttons handling the displayed map
        # ----------------------------------

        self.gui['clim'] = mp_wdgt.TextBox(self.fig.add_axes([0.88, 0.06, 0.11, 0.04]),
                                           label='adjust colorbar: ', textalignment='center',
                                           initial='{0:.3g}, {1:.3g}'.format(*self.pid.get_clim()))
        self.gui['clim'].on_submit(self._set_clim_)
        self.gui['cmap'] = mp_wdgt.TextBox(self.fig.add_axes([0.88, 0.01, 0.11, 0.04]),
                                           label='colormap: ', textalignment='center',
                                           initial=self.pid.get_cmap().name)
        self.gui['cmap'].on_submit(self._set_cmap_)

        self.gui['home'] = mp_wdgt.Button(self.fig.add_axes([0.20, 0.84, 0.08, 0.05]),
                                          'reset axis', color='turquoise', hovercolor='paleturquoise')
        self.gui['home'].on_clicked(self._ResetAxis_)


        # Subfigure for editing new variable properties
        # ---------------------------------------------

        self.subfig = self.fig.add_subfigure(GridSpec(4, 4)[2, 1])
        self.subfig.set_linewidth(2)
        self.subfig.set_edgecolor('maroon')
        self.subfig.set_facecolor('linen')
        self.subfig.suptitle('New variable', fontweight='bold', fontsize=16)

        # Widgets associated to the subfigure
        self.subgui = {}
        self.subgui['varname'] = mp_wdgt.TextBox(self.subfig.add_axes([.25, .7, .5, .12]), 'name: ')
        self.subgui['vartype'] = mp_wdgt.RadioButtons(self.subfig.add_axes([.1, .1, .35, .4]), ['int', 'float', 'double'], active=1)
        self.subgui['vartype'].ax.set_title('data type')
        self.subgui['confirm'] = mp_wdgt.Button(self.subfig.add_axes([.55, .15, .35, .3]), 'CONFIRM')
        self.subgui['confirm'].on_clicked(self._complete_new_var_creation)

        # attribute storing array during its creation
        self._new_array = None

        # Hide the subfigure
        self.set_subfig_active(False)


        # DRAWING
        if show:
            plt.show()



    def set_subfig_active(self, status: bool):
        '''
        Method to make visible (active, status=True) or hide (status=False)
        the new variable editing subfigure
        '''
        zorder = 10 if status else -10
        #
        self.subfig.set_zorder(zorder)
        self.subfig.set_visible(status)
        for obj in self.subgui.values():
            obj.ax.set_visible(status)
            if hasattr(obj, 'circles'):
                for c in obj.circles:
                    c.set_visible(status)


    # Class methods to connect to widgets
    # ===================================


    def _RecSel_Updte_(self, eclick, erelease):
        '''
        Method to update editing mask from rectangle selector
        '''
        i0, j0 = int(eclick.xdata), int(eclick.ydata)
        i1, j1 = int(erelease.xdata)+1, int(erelease.ydata)+1
        if self.selecting:
            self.editmask[j0:j1,i0:i1] = 1
        else:
            self.editmask[j0:j1,i0:i1] = 0
            self.editmask.mask[j0:j1,i0:i1] = True

        self.mpid.set_array(self.editmask)
        plt.draw()


    def _PolSel_Updte_(self, event):
        '''
        Method to update editing mask from polygon or lasso selector
        '''

        pth = mp_Path(event)

        bbox = (pth.vertices.min(0), pth.vertices.max(0))
        i0, j0, i1, j1 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[1][0])+1, int(bbox[1][1])+1
        slc = (slice(j0, j1), slice(i0, i1))

        msk = pth.contains_points(np.array([self.nav_xi[slc].ravel(), self.nav_yi[slc].ravel()]\
                                          ).transpose()).reshape((j1-j0,i1-i0))

        if self.selecting:
            msk = np.logical_or(~self.editmask[slc].mask, msk)
        else:
            msk = np.logical_and(~self.editmask[slc].mask, ~msk)

        self.editmask[slc] = np.ma.masked_where(~msk, msk.astype('int8'))

        self.mpid.set_array(self.editmask)
        plt.draw()


    def _Swtch_Selector_(self, event):
        '''
        Method to switch selecting method ("selector")
        '''
        self.gui['RecSel'].set_active(event=='rectangle selector')
        self.gui['PolSel'].set_active(event=='polygon selector')
        self.gui['LasSel'].set_active(event=='lasso selector')
        plt.draw()


    def _Swtch_SelUnsel_(self, event):
        '''
        Method to switch selecting/unselecting mode
        '''
        self.selecting = (event=='select')
        plt.draw()


    def _Clear_EditMask_(self, event):
        '''
        Method to reset the editing mask to "0"
        '''
        self.editmask.data[:] = 0
        self.editmask.mask[:] = True
        self.mpid.set_array(self.editmask)
        plt.draw()


    def _Edit_Sel_(self, event):
        '''
        Method to trigger data editing (from current mask)
        '''
        # Make a copy of selected area in current variable
        self._data_cache = self.current_array[self.editmask.data.astype(bool)].copy()
        self.gui['value'].ax.set_visible(True)
        plt.draw()


    def _Set_EditWhere_(self, event):
        '''
        Method to set where to edit the data (unmasked values, masked values, or both)
        '''
        self.data_edit_props['where'] = event
        plt.draw()


    def _Set_EditHow_(self, event):
        '''
        Method to set how to edit the data (put an absolute value, a value relative to
        the current value, or a precentage of the current value)
        '''
        if self.data_edit_props['where'] != 'unmasked data':
            if event != 'absolute':
                print('Error: masked data can only be modified with absolute value')
                event = 'absolute'
                self.gui['edit how'].value_selected = 'absolute'
        #
        self.data_edit_props['how'] = event
        plt.draw()


    def _Set_Edit_Value_(self, event):
        '''
        Method to modify the selected area of the current variable,
        according to the value and method specified by the user.
        '''

        # token value that means "fill-value":
        if event.lower() in ['mask', 'masked', 'fill', 'fillval', 'fillvalue', '_fillvalue', 'nan']:
            x = '_FillValue'
        else:
            x = float(event)

        if self.data_edit_props['where'] == 'unmasked data':
            msk = np.logical_and(~self.editmask.mask, ~self.current_array.mask)
        elif self.data_edit_props['where'] == 'masked data':
            msk = np.logical_and(~self.editmask.mask, self.current_array.mask)
        else: # -> self.data_edit_props['where'] == 'both'
            msk = ~self.editmask.mask

        if x == '_FillValue':
            try:
                self.current_array.data[msk] = self.current_array.fill_value
            except AttributeError:
                pass
            #
            self.current_array = np.ma.masked_where(msk, self.current_array, copy=False)

        else:
            if self.data_edit_props['how'] == 'absolute':
                self.current_array.data[msk] = x
            elif self.data_edit_props['how'] == 'delta':
                self.current_array.data[msk] = (self.current_array.data[msk] + x).astype(self.current_array.dtype)
            else: # -> self.data_edit_props['how'] == '%'
                self.current_array.data[msk] = ((x/100)*self.current_array.data[msk]).astype(self.current_array.dtype)
            #
            if np.size(self.current_array.mask) != 1:
                self.current_array.mask[msk] = False

        self.pid.set_array(self.current_array)
        for key in ('confirm', 'cancel'):
            self.gui[key].ax.set_visible(True)

        plt.draw()


    def _Confirm_Edit_(self, event):
        '''
        Method to confirm data editing (=> delete cache and selecting area)
        '''
        self.editmask.data[:] = 0
        self.editmask.mask[:] = True
        self.mpid.set_array(self.editmask)
        for key in ('value', 'confirm', 'cancel'):
            self.gui[key].ax.set_visible(False)

        plt.draw()


    def _Cancel_Edit_(self, event):
        '''
        Method to cancel data editing (=> restore cache, delete it and delete selecting area)
        '''
        self.current_array[self.editmask.data.astype(bool)] = self._data_cache
        self.pid.set_array(self.current_array)
        self._Confirm_Edit_(event)


    def _Iter_NextVar_(self, event):
        '''
        Method to switch to the next 2D variable to display
        '''
        self._ivar = (self._ivar + 1) % len(self.arrays)
        self.current_var = self.variables[self._ivar]
        self.current_array = self.arrays[self._ivar]
        self.current_clim = self.list_clim[self._ivar]
        self.current_cmap = self.list_cmap[self._ivar]
        self.pid.set_array(self.current_array)
        self.pid.set_clim(*self.current_clim)
        self.pid.set_cmap(self.current_cmap)
        self.gui['varnum'].set_text('{0:}/{1:}'.format(self._ivar+1, len(self.arrays)))
        if hasattr(self.current_var, 'name'):
            self.ax.set_title(self.current_var.name)

        plt.draw()


    def _Iter_PrevVar_(self, event):
        '''
        Method to switch to the previous 2D variable to display
        '''
        self._ivar = (self._ivar - 1) % len(self.arrays)
        self.current_var = self.variables[self._ivar]
        self.current_array = self.arrays[self._ivar]
        self.current_clim = self.list_clim[self._ivar]
        self.current_cmap = self.list_cmap[self._ivar]
        self.pid.set_array(self.current_array)
        self.pid.set_clim(*self.current_clim)
        self.pid.set_cmap(self.current_cmap)
        self.gui['varnum'].set_text('{0:}/{1:}'.format(self._ivar+1, len(self.arrays)))
        if hasattr(self.current_var, 'name'):
            self.ax.set_title(self.current_var.name)

        plt.draw()


    def _Add_NewVar_(self, event):
        '''
        Method to create and add to the list a new blank variable
        (zero-filled array, with same mask)
        '''
        # Create new array
        self._new_array = np.ma.masked_where(self.current_array.mask, np.zeros(self.shape, dtype='float32'), copy=True)
        # Make new var editing subfigure active
        self.set_subfig_active(True)
        plt.draw()


    def _Add_CopyVar_(self, event):
        '''
        Method to duplicate the current variable and add it to the list
        '''
        # Create new array
        self._new_array = self.current_array.copy()
        # Make new var editing subfigure active
        self.set_subfig_active(True)
        plt.draw()


    def _complete_new_var_creation(self, event):
        '''
        Private method to complete the creation of the new variable.
        The field (array) must be already created and stored in the attribute "_new_array".
        This method is meant to be called solely by the "confirm" variable creation button.
        '''

        # Get info from subfigure editing widgets
        name  = self.subgui['varname'].text
        dtype = self.subgui['vartype'].value_selected

        if name == '':
            name = 'new_var_{:}'.format(len(self.variables)+1)

        if dtype == 'double':
            dtype = 'float64'
        else: # dtype == 'int' or 'float'
            dtype = dtype+'32'

        # Update current array type
        self._new_array = self._new_array.astype(dtype)


        # Create new variable (depending on current variable type)
        # ........................................................

        try:

            if isinstance(self.current_var, ncGriddedVar):

                newvar = ncGriddedVar(name, datatype=dtype,
                                      replicate_var=self.current_var, link_to=True,
                                      value=self._new_array)
                newvar.add_to_dataset()

            elif isinstance(self.current_var, netCDF4.Variable):
                
                ds = self.current_var.group()
                #
                # Fill-value
                fill_value=None
                for fval in ['missing_value', 'fill_value', '_FillValue']:
                    if hasattr(self.current_var, fval):
                        fill_value = getattr(self.current_var, fval)
                        break
                #
                # Variable definition
                newvar = ds.createVariable('new_var_{:}'.format(len(self.variables)+1),
                                        datatype=dtype, dimensions=self.current_var.dimensions,
                                        fill_value=fill_value)
                #
                # Copy variable attributes
                for att in self.current_var.ncattrs():
                    if att not in ['name', 'missing_value', 'fill_value', '_FillValue']:
                        ds.variables[self.current_var].setncattr(att, self.current_var.getncattr(att))
                #
                # Copy variable data
                newvar[:] = self._new_array

            else:

                print('warning: numpy-array not linked to a file or dataset')
                newvar = self._new_array

            clim = (self._new_array.min(), self._new_array.max())
            if clim[0] == clim[1]:
                clim = (clim[0]-1, clim[0]+1)

            # Update main information
            self.variables.append(newvar)
            self.arrays.append(self._new_array)
            self.list_clim.append(clim)
            self.list_cmap.append(self.current_cmap)
            self._ivar = len(self.arrays) - 1
            self.current_var = self.variables[self._ivar]
            self.current_array = self.arrays[self._ivar]
            self.current_clim = self.list_clim[self._ivar]
            self.current_cmap = self.list_cmap[self._ivar]
            self.pid.set_array(self.current_array)
            self.pid.set_clim(self.current_clim)
            self.gui['varnum'].set_text('{0:}/{1:}'.format(self._ivar+1, len(self.arrays)))
            if hasattr(newvar, 'name'):
                self.ax.set_title(newvar.name)


        finally:

            # Hide new var editing subfigure active
            self.set_subfig_active(False)
            
            # delete temporary array
            self._new_array = None

            plt.draw()


    def _set_clim_(self, event):
        '''
        Method to modify the bounds of the colorbar
        '''
        clim = tuple(float(s) for s in event.split(','))
        self.list_clim[self._ivar] = clim
        self.pid.set_clim(*clim)
        plt.draw()


    def _set_cmap_(self, event):
        '''
        Method to modify the bounds of the colorbar
        '''
        self.pid.set_cmap(event)
        self.list_cmap[self._ivar] = event
        plt.draw()


    def _ResetAxis_(self, event):
        '''
        Method to get back to initial axis bounds (xlim and ylim)
        '''
        self.ax.set_xlim((0, self.shape[1]))
        self.ax.set_ylim((0, self.shape[0]))
        plt.draw()


    def _write_to_file_(self, event):
        '''
        Save all data in corresponding file
        '''
        for var, field in zip(self.variables, self.arrays):
            var[:] = field
            # assert that ncGriddedVar objects are properly written in files
            if isinstance(var, ncGriddedVar):
                if var._value is not None: # => var's data is detached from the file
                    var.save_in_file()


