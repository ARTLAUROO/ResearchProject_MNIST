import time

class ExperimentID:
  """
  Class that identifies uniquely an experiment. It is constructed out of the
  most important settings used for an experiment and the time it is started.
  """
  conv = [32, 64]
  full = [512]

  batch_size = 100
  n_epochs = 10

  day = None
  month = None
  year = None
  hour = None
  minute = None
  second = None

  def init_string(self, string):
    """
    Initialise all settings from a string with the following format:
    C-32-64-F-1024_B-100-E-1_05-Dec-2016_11-59-33
    """
    layers, settings, date, time = string.split('_')

    layers = layers.split('-')
    f_idx = layers.index('F')
    self.conv = layers[1:f_idx]
    self.full = layers[f_idx+1:]

    settings = settings.split('-')
    self.batch_size = int(settings[1])
    if settings[3] == 'None':
      self.n_epochs = None
    else:
      self.n_epochs = int(settings[3])

    date = date.split('-')
    self.day = date[0]
    self.month = date[1]
    self.year = date[2]

    time = time.split('-')
    self.hour = time[0]
    self.minute = time[1]
    self.second = time[2]


  def init_settings(self, conv, full, batch_size, n_epochs):
    """
    Initialise from settings used in an experiment and the current time.
    :param conv: list with kernels sizes per conv layer
    :param full: list with number of nodes per conv layer
    :param batch_size: batch_size used in training
    :param n_epochs: trained for this many epochs, can be None to declare
      infinite
    :return:
    """
    self.conv = conv
    self.full = full

    self.batch_size = batch_size
    self.n_epochs = n_epochs

    date_time = time.gmtime()
    self.day = time.strftime('%d', date_time)
    self.month = time.strftime('%b', date_time)
    self.year = time.strftime('%Y', date_time)

    self.hour = time.strftime('%H', date_time)
    self.minute = time.strftime('%M', date_time)
    self.second = time.strftime('%S', date_time)

  def __repr__(self):
    """
    Returns string representation of self in the form of:
    C-32-64-F-1024_B-100-E-1_05-Dec-2016_11-59-33
    """
    string = 'C'
    for c in self.conv:
      string += '-{}'.format(c)

    string += '-F'
    for f in self.full:
      string += '-{}'.format(f)

    string += '_B-{}-E-{}'.format(self.batch_size, self.n_epochs)

    string += '_{}-{}-{}_{}-{}-{}'.format(self.day,
                                          self.month,
                                          self.year,
                                          self.hour,
                                          self.minute,
                                          self.second)
    return string