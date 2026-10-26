import json
import os
from threading import Event
import time
import unittest
from unittest.mock import MagicMock, patch

import requests
from streamlit.testing.v1 import AppTest

from euclid_umap_explorer import components, sled, sled_ui
from euclid_umap_explorer.service_icons import ALADIN_ICON_SRC, SLED_ICON_SRC, aladin_icon_link


def public_record(**overrides):
    return {
        "name": "Example lens", "ra": "100.0", "dec": "30.0",
        "access_level": "PUB", "flag": "CONFIRMED", "papers": [{
            "year": 2025, "first_author": "Example", "title": "Lens study",
            "cite_as": "2025Example",
        }], **overrides,
    }


def response_for(records):
    return {"errors": [], "lenses": records}


CREDS = {"SLED_USERNAME": "test-user", "SLED_PASSWORD": "not-a-real-password"}


class SledNetworkTests(unittest.TestCase):
    def setUp(self):
        self.query = sled.SledQuery(100.0, 30.0)

    def test_public_cone_parameters_and_units(self):
        params = self.query.parameters()
        self.assertEqual(params["management-access_level"], "PUB")
        self.assertAlmostEqual(params["lens-radius"], 5 / 3600)
        self.assertEqual(params["download-related"], ["papers"])
        self.assertNotIn("username", params)
        self.assertIn("management-access_level=PUB", self.query.search_url())

    def test_coordinate_validation(self):
        for args in ((float("nan"), 0), (360, 0), (0, 91), (0, 0, 0), (0, 0, 61)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                sled.SledQuery(*args)

    def test_valid_match_is_not_a_new_escope_label(self):
        result = sled.parse_response(response_for([public_record()]), self.query)
        self.assertEqual(result.status, "matched")
        self.assertEqual(result.matches[0].flag, "CONFIRMED")
        self.assertAlmostEqual(result.matches[0].separation_arcsec, 0)
        self.assertEqual(result.matches[0].papers[0].year, "2025")
        self.assertFalse(hasattr(result, "is_lens"))
        self.assertNotIn(CREDS["SLED_PASSWORD"], repr(result))

    def test_empty_response_is_a_miss_but_invalid_responses_are_not(self):
        self.assertEqual(sled.parse_response(response_for([]), self.query).status, "no_match")
        for payload in ({}, [], {"lenses": []}, {"errors": ["bad query"], "lenses": []},
                        {"errors": [], "lenses": {}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                sled.parse_response(payload, self.query)

    def test_private_unknown_access_and_out_of_cone_records_fail_closed(self):
        for record in (public_record(access_level="PRI"), public_record(access_level=None),
                       public_record(ra=110), public_record(ra="NaN")):
            with self.subTest(record=record), self.assertRaises(ValueError):
                sled.parse_response(response_for([record]), self.query)

    def test_multiple_matches_are_sorted_by_angular_distance(self):
        result = sled.parse_response(response_for([
            public_record(name="further", dec=30 + 3 / 3600),
            public_record(name="closer", dec=30 + 1 / 3600, flag="CANDIDATE"),
        ]), self.query)
        self.assertEqual([m.name for m in result.matches], ["closer", "further"])
        self.assertAlmostEqual(result.matches[0].separation_arcsec, 1.0, places=5)
        self.assertIn("nearby", sled_ui.result_presentation(result)[0])

    def test_spherical_matching_across_ra_zero(self):
        result = sled.parse_response(response_for([
            public_record(ra=359.9998, dec=0)
        ]), sled.SledQuery(.0002, 0))
        self.assertAlmostEqual(result.matches[0].separation_arcsec, 1.44, places=5)

    def _post_result(self, *, body=None, status=200, exception=None, headers=None):
        response = MagicMock()
        response.status_code = status
        response.headers = headers or {}
        response.iter_content.return_value = [json.dumps(body).encode()]
        context = MagicMock()
        context.__enter__.return_value = response
        with patch.dict(os.environ, CREDS), patch.object(
            sled.requests, "post", return_value=context, side_effect=exception
        ) as post:
            result = sled.query_sled(self.query)
        return result, post, context

    def test_http_request_is_bounded_read_only_and_does_not_follow_redirects(self):
        result, post, context = self._post_result(body=response_for([]))
        self.assertEqual(result.status, "no_match")
        self.assertEqual(post.call_args.args, (sled.SLED_ENDPOINT,))
        self.assertEqual(post.call_args.kwargs["timeout"], (2, 4))
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertEqual(post.call_args.kwargs["auth"].username, CREDS["SLED_USERNAME"])
        context.__exit__.assert_called_once()

    def test_failure_modes_never_become_false_misses(self):
        for status in (301, 401, 403, 429, 500):
            with self.subTest(status=status):
                result, _, _ = self._post_result(status=status)
                self.assertEqual(result.status, "error")
        for exception, expected in (
            (requests.exceptions.Timeout("secret"), "timeout"),
            (requests.exceptions.ConnectionError("secret"), "error"),
        ):
            result, _, _ = self._post_result(exception=exception)
            self.assertEqual(result.status, expected)
            self.assertNotIn("secret", result.message)
        result, _, _ = self._post_result(body={"detail": "provider error"})
        self.assertEqual(result.status, "error")

    def test_oversized_and_private_responses_are_rejected(self):
        result, _, _ = self._post_result(body="x" * sled.MAX_RESPONSE_BYTES)
        self.assertEqual(result.status, "error")
        result, _, _ = self._post_result(body=response_for([public_record(access_level="PRI")]))
        self.assertEqual(result.status, "error")
        self.assertFalse(result.matches)

    def test_cloudflare_challenge_is_not_misreported_as_invalid_credentials(self):
        result, _, _ = self._post_result(status=403, headers={"cf-mitigated": "challenge"})
        self.assertEqual(result.status, "error")
        self.assertIn("access protection", result.message)
        self.assertNotIn("credentials", result.message)

    def test_missing_credentials_do_not_send_a_request(self):
        with patch.dict(os.environ, {"SLED_USERNAME": "", "SLED_PASSWORD": ""}), \
                patch.object(sled.requests, "post") as post:
            self.assertEqual(sled.query_sled(self.query).status, "unavailable")
            post.assert_not_called()


class SledCacheTests(unittest.TestCase):
    def setUp(self):
        self.credentials = patch.dict(os.environ, CREDS)
        self.credentials.start()
        self.addCleanup(self.credentials.stop)
        self.now = 0.0
        self.query = sled.SledQuery(100, 30)
        self.release = Event()
        self.addCleanup(self.release.set)

    def client(self, lookup, **kwargs):
        client = sled.SledClient(lookup=lookup, clock=lambda: self.now, **kwargs)
        self.addCleanup(client.close)
        return client

    def wait_result(self, client, query=None):
        query = query or self.query
        for _ in range(200):
            result = client.get(query)
            if result and result.status != "pending":
                return result
            time.sleep(.005)
        self.fail("Worker did not finish")

    def test_no_implicit_query_and_deduplicated_nonblocking_requests(self):
        started = Event()
        calls = []

        def lookup(query):
            calls.append(query)
            started.set()
            self.release.wait(2)
            return sled.SledResult("no_match", checked_at="now")

        client = self.client(lookup)
        self.assertIsNone(client.get(self.query))
        self.assertFalse(calls)
        before = time.monotonic()
        self.assertEqual(client.request(self.query).status, "pending")
        self.assertTrue(started.wait(.5))
        self.assertEqual(client.request(self.query).status, "pending")
        self.assertEqual(client.get(self.query).status, "pending")
        self.assertLess(time.monotonic() - before, .5)
        self.assertEqual(len(calls), 1)
        self.release.set()
        self.assertEqual(self.wait_result(client).status, "no_match")
        self.assertEqual(client.request(self.query).status, "no_match")
        self.assertEqual(len(calls), 1)

    def test_timeout_is_terminal_and_late_result_does_not_replace_it(self):
        done = Event()

        def lookup(_query):
            self.release.wait(2)
            done.set()
            return sled.SledResult("no_match")

        client = self.client(lookup, workers=1)
        client.request(self.query)
        self.now = sled.LOOKUP_TIMEOUT_SECONDS + .1
        self.assertEqual(client.get(self.query).status, "timeout")
        # The running socket is not replaced by another thread after UI timeout.
        self.assertEqual(client.request(self.query, refresh=True).status, "timeout")
        self.assertEqual(client.request(sled.SledQuery(101, 30)).status, "busy")
        self.release.set()
        self.assertTrue(done.wait(.5))
        self.assertEqual(self.wait_result(client).status, "timeout")
        time.sleep(.02)
        self.assertEqual(client.get(self.query).status, "timeout")

    def test_bounded_workers_and_no_growing_queue(self):
        client = self.client(lambda _q: (self.release.wait(2), sled.SledResult("no_match"))[1], workers=2)
        self.assertEqual(client.request(self.query).status, "pending")
        self.assertEqual(client.request(sled.SledQuery(101, 30)).status, "pending")
        self.assertEqual(client.request(sled.SledQuery(102, 30)).status, "busy")
        self.assertEqual(len(client._pending), 2)
        self.release.set()

    def test_success_empty_and_error_cache_lifetimes(self):
        for status, ttl in (("matched", sled.MATCH_CACHE_SECONDS),
                            ("no_match", sled.EMPTY_CACHE_SECONDS),
                            ("error", sled.ERROR_CACHE_SECONDS)):
            with self.subTest(status=status):
                self.now = 0
                client = self.client(lambda _q: sled.SledResult(status))
                client.request(self.query)
                self.wait_result(client)
                self.now = ttl - .1
                self.assertEqual(client.get(self.query).status, status)
                self.now = ttl + .1
                self.assertIsNone(client.get(self.query))

    def test_cache_expiry_is_based_on_completion_not_first_late_poll(self):
        completed = Event()

        def lookup(_query):
            completed.set()
            return sled.SledResult("no_match")

        client = self.client(lookup)
        client.request(self.query)
        self.assertTrue(completed.wait(.5))
        time.sleep(.02)
        self.now = sled.EMPTY_CACHE_SECONDS + 1
        self.assertIsNone(client.get(self.query))

    def test_refresh_cooldown_and_cache_key_include_coordinates_and_radius(self):
        calls = []
        client = self.client(lambda q: (calls.append(q), sled.SledResult("no_match"))[1])
        client.request(self.query)
        self.wait_result(client)
        client.request(self.query, refresh=True)
        self.assertEqual(len(calls), 1)
        self.now = 6
        client.request(self.query, refresh=True)
        self.wait_result(client)
        for query in (sled.SledQuery(101, 30), sled.SledQuery(100, 30, 10)):
            client.request(query)
            self.wait_result(client, query)
        self.assertEqual(len(calls), 4)

    def test_lru_cache_bound_and_sanitized_worker_errors(self):
        def lookup(query):
            if query.ra == 100:
                raise RuntimeError("password=SECRET")
            return sled.SledResult("no_match")

        client = self.client(lookup, max_entries=2)
        client.request(self.query)
        result = self.wait_result(client)
        self.assertEqual(result.status, "error")
        self.assertNotIn("SECRET", repr(result))
        for ra in (101, 102):
            query = sled.SledQuery(ra, 30)
            client.request(query)
            self.wait_result(client, query)
        self.assertEqual(len(client._cache), 2)
        self.assertIsNone(client.get(self.query))


class SledUITests(unittest.TestCase):
    def setUp(self):
        enabled = patch.object(sled_ui, "ENABLE_SLED_CHECK", True)
        enabled.start()
        self.addCleanup(enabled.stop)

    SCRIPT = """
import streamlit as st
from euclid_umap_explorer.sled_ui import render_sled_check
obj = st.selectbox('Object', ['1', '2'])
st.session_state['expensive_runs'] = st.session_state.get('expensive_runs', 0) + 1
render_sled_check(obj, 100.0 if obj == '1' else 101.0, 30.0, key='test')
st.button('Unrelated action')
"""

    def test_explicit_query_pending_result_and_object_changes(self):
        client = MagicMock()
        client.get.return_value = None
        client.request.return_value = sled.SledResult("pending")
        with patch.dict(os.environ, CREDS), patch.object(sled_ui, "sled_client", return_value=client):
            app = AppTest.from_string(self.SCRIPT).run()
            self.assertFalse(app.exception)
            client.request.assert_not_called()
            next(b for b in app.button if b.label == sled_ui.SLED_BUTTON_LABEL).click().run()
            self.assertFalse(app.exception)
            client.request.assert_called_once_with(sled.SledQuery(100, 30), refresh=False)
            client.get.return_value = sled.SledResult("pending")
            app.run()
            pending_button = next(b for b in app.button if b.label == sled_ui.SLED_BUTTON_LABEL)
            self.assertTrue(pending_button.disabled)
            self.assertNotIn("Click to check or refresh", pending_button.proto.help)
            self.assertFalse(next(b for b in app.button if b.label == "Unrelated action").disabled)
            client.get.return_value = sled.parse_response(response_for([public_record()]), sled.SledQuery(100, 30))
            app.run()
            self.assertFalse(app.exception)
            self.assertIn("Confirmed lens nearby", " ".join(m.value for m in app.markdown))
            self.assertEqual(len(app.dataframe), 2)
            client.get.return_value = None
            app.selectbox[0].set_value("2").run()
            self.assertFalse(app.exception)
            self.assertNotIn("Confirmed lens nearby", " ".join(m.value for m in app.markdown))
            client.request.assert_called_once()

    def test_credentials_and_coordinates_disable_query_without_crashing(self):
        for creds, ra in (({}, "100.0"), (CREDS, "None")):
            with self.subTest(creds=bool(creds)), patch.dict(os.environ, {
                "SLED_USERNAME": "", "SLED_PASSWORD": "", **creds
            }), patch.object(sled_ui, "sled_client") as factory:
                factory.return_value.get.return_value = None
                app = AppTest.from_string(
                    "from euclid_umap_explorer.sled_ui import render_sled_check\n"
                    f"render_sled_check('1', {ra}, 30.0, key='test')"
                ).run()
                self.assertFalse(app.exception)
                self.assertTrue(app.button[0].disabled)
                self.assertNotIn("Click to check or refresh", app.button[0].proto.help)
                factory.return_value.request.assert_not_called()

    def test_escaped_provider_text_and_distinct_miss_timeout_error_states(self):
        for result in (sled.SledResult("no_match", checked_at="now"),
                       sled.SledResult("timeout", checked_at="now", message="Time limit"),
                       sled.SledResult("error", checked_at="now", message="Failed"),
                       sled.parse_response(response_for([public_record(name='<script>alert(1)</script>')]), sled.SledQuery(100, 30))):
            with self.subTest(status=result.status), patch.dict(os.environ, CREDS), \
                    patch.object(sled_ui, "sled_client") as factory:
                factory.return_value.get.return_value = result
                app = AppTest.from_string(self.SCRIPT).run()
                self.assertFalse(app.exception)
                text = " ".join(m.value for m in app.markdown)
                self.assertNotIn("<script>", text)
                self.assertIn(sled_ui.result_presentation(result)[0], text)
                factory.return_value.request.assert_not_called()

    def test_selected_object_only_exposes_sled_check_for_unknowns(self):
        for labelled in (False, True):
            app = AppTest.from_string(f"""
import pandas as pd
from euclid_umap_explorer.components import show_object_details
show_object_details(pd.Series({{'object_id': 'sled-ui-test', 'is_lens': {labelled},
                               'lens_grade': 'A' if {labelled} else ''}}), key='test')
""")
            with patch.object(components, "selected_umap_cutout_path", return_value=None), \
                    patch.object(components, "lens_image_path", return_value=None), \
                    patch.object(components, "ENABLE_SLED_CHECK", True), \
                    patch.object(components, "object_sky_coordinates", return_value=(100, 30)), \
                    patch.object(components, "render_sled_check") as render:
                app.run()
                self.assertFalse(app.exception)
                if labelled:
                    render.assert_not_called()
                else:
                    render.assert_called_once_with('sled-ui-test', 100, 30, key='test')

    def test_disabled_sled_is_hidden_and_does_not_create_a_client(self):
        with patch.object(sled_ui, "ENABLE_SLED_CHECK", False), \
                patch.object(sled_ui, "sled_client") as factory:
            app = AppTest.from_string(self.SCRIPT).run()
            self.assertFalse(app.exception)
            self.assertEqual([button.label for button in app.button], ["Unrelated action"])
            self.assertFalse(app.markdown)
            factory.assert_not_called()

    def test_service_logos_are_bundled_and_aladin_has_a_tooltip_not_visible_text(self):
        from html.parser import HTMLParser

        class LinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tags, self.text = [], []

            def handle_starttag(self, tag, attrs):
                self.tags.append((tag, dict(attrs)))

            def handle_data(self, data):
                self.text.append(data)

        for image in (ALADIN_ICON_SRC, SLED_ICON_SRC):
            self.assertTrue(image.startswith("data:image/png;base64,"))
        parser = LinkParser()
        parser.feed(aladin_icon_link('https://example.org/?x="unsafe"&y=1'))
        anchor = next(attrs for tag, attrs in parser.tags if tag == "a")
        self.assertEqual(anchor["title"], "View in Aladin")
        self.assertEqual(anchor["aria-label"], "View in Aladin")
        self.assertEqual(anchor["target"], "_blank")
        self.assertEqual(anchor["href"], 'https://example.org/?x="unsafe"&y=1')
        self.assertIn("background:#e5e7eb", anchor["style"])
        self.assertIn("border:1px solid #cbd0d6", anchor["style"])
        self.assertIn("flex:none", anchor["style"])
        self.assertIn("width:52px;height:34px", anchor["style"])
        self.assertEqual("".join(parser.text), "")
        self.assertTrue(any(tag == "img" for tag, _ in parser.tags))
        self.assertIn("![Check SLED](data:image/png;base64,", sled_ui.SLED_BUTTON_LABEL)


if __name__ == "__main__":
    unittest.main()
