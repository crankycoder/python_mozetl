# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

from random import randint


class XValidator:
    """
    The xvalidator class will take in a dictionary of the form:

        client_id => {
            "geo_city": profile_data.get("city", ''),
            "subsession_length": profile_data.get("subsession_length", 0),
            "locale": profile_data.get('locale', ''),
            "os": profile_data.get("os", ''),
            "installed_addons": addon_ids,
            "disabled_addons_ids": profile_data.get("disabled_addons_ids", []),
            "bookmark_count": profile_data.get("places_bookmarks_count", 0),
            "tab_open_count": profile_data.get("scalar_parent_browser_engagement_tab_open_event_count", 0),
            "total_uri": profile_data.get("scalar_parent_browser_engagement_total_uri_count", 0),
            "unique_tlds": profile_data.get("scalar_parent_browser_engagement_unique_domains_count", 0),
        }

    We want to splice the inbound data into N number of folds to do a
    standard cross validation where one fold is used to test the
    training of the model based on the other N-1 folds of data.

    The additional constraint we have is that the test dataset must
    mask some of the addons in the 'installed_addons' list.  This is
    so that the recommender has a chance to 'fill' in the masked
    addons.
    """

    def __init__(self, n_folds, addons_minsize):
        assert n_folds > 1
        self._n_folds = n_folds
        self._addons_minsize = addons_minsize

    def cross_validation_split(self, dataset):
        """Split a dataset into k folds
        """
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / self._n_folds)
        for fold_i in range(self._n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randint(0, len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)

        return dataset_split

    def mask_addons(self, dataslice):
        # Splice the dataset_split into a list of training splits and
        # a testing split

        # Note that each of the client_data blobs in the dataset
        # does *not* contain a client ID.  This is ok for us, as we
        # only care to reconcile the masked installed addons with the
        # predicted addons.  We don't care about reconciling all the
        # way back to the original client ID

        masked_addons_by_clientid = {}
        for idx, client_data in enumerate(dataslice):
            client_data['addon_mask_id'] = idx

            # use a random selection of addons
            installed_addon_set = list(client_data['installed_addons'])

            keep_set = set()
            for i in range(self._addons_minsize):
                idx = randint(0, len(installed_addon_set))
                keep_set.add(installed_addon_set.pop(idx))
            masked_addon_set = set(installed_addon_set) - keep_set

            masked_addons_by_clientid[idx] = list(masked_addon_set)
            client_data['installed_addons'] = list(keep_set)

        return (dataslice, masked_addons_by_clientid)

    def check_predicted_addons(self,
                               client_data,
                               predicted_addons,
                               masked_addons_by_clientid):
        """
        Check the predicted addons for a singular client.

        Return a 3-tuple of :

        (prediction_accuracy_rate, expected_addons_set, correctly_predicted_addon_set)
        """
        addon_mask_id = client_data['addon_mask_id']

        expected_addons = set(masked_addons_by_clientid[addon_mask_id])
        predicted_set = set(predicted_addons)
        match_set = expected_addons.intersect(predicted_set)

        prediction_rate = len(match_set) / len(expected_addons) * 1.0
        return prediction_rate, expected_addons, match_set
