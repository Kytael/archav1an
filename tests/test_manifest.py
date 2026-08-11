from tools.archive_batch.manifest import Clip, parse_manifest, order_clips

ONE_STREAM = "SetA/2001/event-a/MVI_4743.MOV\t28340332\t24000/1001,81\t3.378375\t"
TWO_STREAM = "SetA/2003/event-c/MVI_0068.MOV\t424279756\t30000/1001,3505\t30000/1001,40\t116.883433\t"


def test_parse_single_video_stream_row():
    clips = parse_manifest(ONE_STREAM + "\n")
    assert len(clips) == 1
    c = clips[0]
    assert c.src == "SetA/2001/event-a/MVI_4743.MOV"
    assert c.rel_dir == "SetA/2001/event-a"
    assert c.stem == "MVI_4743"
    assert c.size == 28340332
    assert c.frames == 81


def test_parse_two_video_stream_row_takes_the_first_stream():
    """An embedded thumbnail adds a second rate column; 40 is its frame
    count, not the video's."""
    clips = parse_manifest(TWO_STREAM + "\n")
    assert clips[0].frames == 3505


def test_parse_skips_blank_lines():
    assert parse_manifest("\n" + ONE_STREAM + "\n\n") == (parse_manifest(ONE_STREAM + "\n")[0],)


def test_order_events_before_practice():
    clips = (
        Clip("SetB/2001/a/x.MOV", "SetB/2001/a", "x", 1, 100),
        Clip("SetA/2001/a/y.MOV", "SetA/2001/a", "y", 1, 100),
    )
    assert [c.stem for c in order_clips(clips)] == ["y", "x"]


def test_order_years_ascending():
    clips = (
        Clip("SetA/2003/a/b.MOV", "SetA/2003/a", "b", 1, 100),
        Clip("SetA/2001/a/a.MOV", "SetA/2001/a", "a", 1, 100),
    )
    assert [c.stem for c in order_clips(clips)] == ["a", "b"]


def test_order_undated_folders_after_year_folders():
    clips = (
        Clip("SetB/routine-two/z.MOV", "SetB/routine-two", "z", 1, 100),
        Clip("SetB/2007/a/w.MOV", "SetB/2007/a", "w", 1, 100),
    )
    assert [c.stem for c in order_clips(clips)] == ["w", "z"]


def test_order_longest_first_within_a_folder():
    clips = (
        Clip("SetA/2001/a/short.MOV", "SetA/2001/a", "short", 1, 100),
        Clip("SetA/2001/a/long.MOV", "SetA/2001/a", "long", 1, 9000),
    )
    assert [c.stem for c in order_clips(clips)] == ["long", "short"]
