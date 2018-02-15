import numpy


def neg_log_sig(log_odds):
    neg_log_odds = [-1.0 * x for x in log_odds]
    e = numpy.exp(neg_log_odds)
    return [numpy.log(1 + f) for f in e if f < (f + 1)]


def cllr(lrs_on_target, lrs_off_target):
    """Compute the log likelihood ratio cost which should be minimized.
    based on Niko Brummer's original implementation:
    Niko Brummer and Johan du Preez, Application-Independent Evaluation of Speaker Detection"
    Computer Speech and Language, 2005
    """
    lrs_on_target = numpy.log(lrs_on_target[~numpy.isnan(lrs_on_target)])
    lrs_off_target = numpy.log(lrs_off_target[~numpy.isnan(lrs_off_target)])

    c1 = numpy.mean(neg_log_sig(lrs_on_target)) / numpy.log(2)
    c2 = numpy.mean(neg_log_sig(-1.0 * lrs_off_target)) / numpy.log(2)
    return (c1 + c2) / 2


def eval_cllr(recommendations_list, unmasked_addons):
    """ A helper function to evaluate the performance of a particular recommendation
    strategy on a client with a set of installed addons that have been patially masked.
    Keyword arguments:
    recommendations_list -- a list of tuples containing (guid, confidence) pairs.
    unmasked_addons -- a list of the true installed addon guids for a test client.
    """
    # Organizer function to extract weights from recommendation list for passing to cllr.
    lrs_on_target_helper = [item[1] for item in recommendations_list if item[0] in unmasked_addons]
    lrs_off_target_helper = [item[1] for item in recommendations_list if item[0] not in unmasked_addons]
    return cllr(lrs_on_target_helper, lrs_off_target_helper)
