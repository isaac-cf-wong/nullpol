import bilby
from bilby.core.utils.introspection import PropertyAccessor


bilby.gw.detector.interferometer.Interferometer.whitened_time_frequency_domain_strain_array = PropertyAccessor('strain_data', 'whitened_time_frequency_domain_strain_array')
bilby.gw.detector.interferometer.Interferometer.whitened_time_frequency_domain_quadrature_strain_array = PropertyAccessor('strain_data', 'whitened_time_frequency_domain_quadrature_strain_array')
bilby.gw.detector.interferometer.Interferometer.Nf = PropertyAccessor('strain_data', 'Nf')
bilby.gw.detector.interferometer.Interferometer.Nt = PropertyAccessor('strain_data', 'Nt')