use std::{path::Path, sync::LazyLock, time::SystemTime};

use anyhow::Result;
use exif::{In, Tag, Value};
use jiff::{
    Span, ToSpan, Zoned,
    civil::{Date, DateTime, Time},
    tz::{Offset, TimeZone},
};
use regex::Regex;

use crate::decode;

#[derive(Debug, Clone, PartialEq)]
pub struct Meta {
    pub width: u32,
    pub height: u32,
    pub exif: Exif,
}

/// The date that --date and --sort date use: when the photo was taken, else when the file last
/// changed. Phone galleries do the same, so screenshots and downloads get a date too.
pub fn date_of(meta: Option<&Meta>, modified: SystemTime) -> Option<DateTime> {
    (meta.and_then(|m| m.exif.taken)).or_else(|| local(modified))
}

/// Wall-clock time in the local time zone, to the second, as in EXIF and in the output. Files
/// with the same second then sort by path.
fn local(time: SystemTime) -> Option<DateTime> {
    let time = Zoned::try_from(time).ok()?.datetime();
    time.with().subsec_nanosecond(0).build().ok()
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exif {
    pub taken: Option<DateTime>,
    pub camera: Option<String>,
    pub gps: bool,
}

/// Half-open interval `lo <= v < hi`. Parsing turns the inclusive CLI bounds into this form.
#[derive(Debug, Clone, PartialEq)]
pub struct Range<T> {
    pub lo: Option<T>,
    pub hi: Option<T>,
}

impl<T: PartialOrd + Copy> Range<T> {
    pub fn contains(&self, v: T) -> bool {
        self.lo.is_none_or(|lo| v >= lo) && self.hi.is_none_or(|hi| v < hi)
    }

    fn parse(s: &str, bound: impl Fn(&str) -> Result<Bound<T>, String>) -> Result<Self, String> {
        let side = |b: &str| (!b.is_empty()).then(|| bound(b)).transpose();
        let (lo, hi) = match s.split_once("..") {
            Some((a, b)) => (side(a)?, side(b)?),
            None => match bound(s)? {
                age @ Bound::Age(_) => (Some(age), None),
                period => (Some(period), Some(period)),
            },
        };
        match (lo, hi) {
            (None, None) => Err("expected A..B, A.., ..B or A".into()),
            (Some(Bound::Age(a)), Some(Bound::Age(b))) if a >= b => {
                Err("ages go oldest first: 1y..1d".into())
            }
            (Some(a), Some(b)) if a.start() >= b.end() => Err("start is after end".into()),
            _ => Ok(Range {
                lo: lo.map(Bound::start),
                hi: hi.map(Bound::end),
            }),
        }
    }
}

/// What one side of a range covers.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Bound<T> {
    /// `[start, end)`, such as one pixel count or a day.
    Period(T, T),
    /// Now minus an age. Alone, it means the time since then.
    Age(T),
}

impl<T: Copy> Bound<T> {
    fn start(self) -> T {
        match self {
            Self::Period(t, _) | Self::Age(t) => t,
        }
    }

    fn end(self) -> T {
        match self {
            Self::Period(_, t) | Self::Age(t) => t,
        }
    }
}

pub fn pixel_range(s: &str) -> Result<Range<u32>, String> {
    Range::parse(s, |p| {
        let n: u32 = p
            .parse()
            .map_err(|_| format!("'{p}' is not a pixel count"))?;
        Ok(Bound::Period(n, n.saturating_add(1)))
    })
}

pub fn time_range(s: &str) -> Result<Range<DateTime>, String> {
    time_range_at(s, &Zoned::now())
}

fn time_range_at(s: &str, now: &Zoned) -> Result<Range<DateTime>, String> {
    Range::parse(s, |b| time_bound(b, now))
}

static AGE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+ ?[a-z]+(?: ?\d+ ?[a-z]+)*(?: ago)?$").unwrap());

static CALENDAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)^
        (?<y>\d{4}) (?: -(?<mo>\d{2}) (?: -(?<d>\d{2}) (?: [Tt\ ](?<h>\d{2})
            (?: :(?<mi>\d{2}) (?: :(?<s>\d{2}) (?: \.(?<f>\d{1,9}) )? )? )?
            (?<z> [Zz] | (?: \+ | (?<minus>-) ) (?<oh>\d{2}) (?: :(?<om>[0-5]\d) )? )?
        )? )? )?$",
    )
    .unwrap()
});

/// The shape of the text picks the kind of bound. The jiff parsers accept too much to try them
/// in turn: "2024" parses as the time 20:24.
fn time_bound(s: &str, now: &Zoned) -> Result<Bound<DateTime>, String> {
    // "2024z" has the shape of an age, but no unit.
    let is_age = |s: &str| AGE.is_match(s) && s.parse::<Span>().is_ok();
    let day = |d: Date| Some(Bound::Period(d.into(), d.tomorrow().ok()?.into()));
    let bound = match s {
        "today" => day(now.date()),
        "yesterday" => now.date().yesterday().ok().and_then(day),
        _ if is_age(s) => s
            .parse::<Span>()
            .and_then(|span| now.checked_sub(span.abs()))
            .ok()
            .map(|t| Bound::Age(t.datetime())),
        _ if s.strip_prefix(['+', '-']).is_some_and(is_age) => {
            return Err("an age has no sign: 7d.. for newer, ..7d for older".into());
        }
        _ if is_age(&s.to_lowercase()) => {
            return Err("units are lowercase: mo for months, m for minutes".into());
        }
        _ => calendar(s, now.time_zone()),
    };
    bound.ok_or_else(|| {
        format!("'{s}' is not a date (2024-06-15T14:30), an age (7d, 1h30m), today or yesterday")
    })
}

/// The period of the last written component. A date with an offset is an instant, so its period
/// becomes wall-clock time in `tz`. A date without one is wall-clock time already.
fn calendar(s: &str, tz: &TimeZone) -> Option<Bound<DateTime>> {
    let c = CALENDAR.captures(s)?;
    let num = |name, default: i8| {
        c.name(name)
            .map_or(default, |m| m.as_str().parse().unwrap())
    };
    let frac = c.name("f").map_or("", |m| m.as_str());
    let date = Date::new(c["y"].parse().unwrap(), num("mo", 1), num("d", 1)).ok()?;
    let nanos = format!("{frac:0<9}").parse().unwrap();
    let start = date.to_datetime(Time::new(num("h", 0), num("mi", 0), num("s", 0), nanos).ok()?);
    let unit = match frac.len() {
        0 => [
            ("s", 1.second()),
            ("mi", 1.minute()),
            ("h", 1.hour()),
            ("d", 1.day()),
            ("mo", 1.month()),
        ]
        .into_iter()
        .find_map(|(name, unit)| c.name(name).map(|_| unit))
        .unwrap_or(1.year()),
        n => 10_i64.pow(9 - n as u32).nanoseconds(),
    };
    let end = start.checked_add(unit).ok()?;
    if c.name("z").is_none() {
        return Some(Bound::Period(start, end));
    }
    let secs = 3600 * i32::from(num("oh", 0)) + 60 * i32::from(num("om", 0));
    let offset = Offset::from_seconds(if c.name("minus").is_some() {
        -secs
    } else {
        secs
    })
    .ok()?;
    // The period keeps its length. Its end in local time can repeat its start, when the clocks
    // go back.
    let start = tz.to_datetime(offset.to_timestamp(start).ok()?);
    Some(Bound::Period(start, start.checked_add(unit).ok()?))
}

#[derive(Default)]
pub struct MetaFilter {
    pub width: Option<Range<u32>>,
    pub height: Option<Range<u32>>,
    pub date: Option<Range<DateTime>>,
    pub camera: Option<Regex>,
    pub gps: Option<bool>,
}

impl MetaFilter {
    pub fn needs_meta(&self) -> bool {
        self.width.is_some()
            || self.height.is_some()
            || self.date.is_some()
            || self.camera.is_some()
            || self.gps.is_some()
    }

    pub fn check(&self, meta: &Meta, modified: SystemTime) -> bool {
        within(&self.width, || Some(meta.width))
            && within(&self.height, || Some(meta.height))
            && within(&self.date, || date_of(Some(meta), modified))
            && self
                .camera
                .as_ref()
                .is_none_or(|re| meta.exif.camera.as_deref().is_some_and(|c| re.is_match(c)))
            && self.gps.is_none_or(|gps| gps == meta.exif.gps)
    }
}

/// An unknown value fails a filter that is set.
fn within<T: PartialOrd + Copy>(
    range: &Option<Range<T>>,
    value: impl FnOnce() -> Option<T>,
) -> bool {
    range
        .as_ref()
        .is_none_or(|r| value().is_some_and(|v| r.contains(v)))
}

pub fn read(path: &Path) -> Result<Meta> {
    let header = decode::header(path)?;
    let (width, height) = header.size();
    Ok(Meta {
        width,
        height,
        exif: header.exif.as_ref().map(exif_fields).unwrap_or_default(),
    })
}

fn exif_fields(exif: &exif::Exif) -> Exif {
    let text = |tag: Tag| match &exif.get_field(tag, In::PRIMARY)?.value {
        Value::Ascii(v) => Some(
            String::from_utf8_lossy(v.first()?)
                .trim_matches(|c: char| c == '\0' || c.is_whitespace())
                .to_string(),
        )
        .filter(|s| !s.is_empty()),
        _ => None,
    };
    // A photo editor writes its export date to DateTime, so DateTime comes last.
    let taken = [Tag::DateTimeOriginal, Tag::DateTimeDigitized, Tag::DateTime]
        .into_iter()
        .find_map(|tag| {
            let t = exif::DateTime::from_ascii(text(tag)?.as_bytes()).ok()?;
            let [mo, d, h, mi, s] = [t.month, t.day, t.hour, t.minute, t.second].map(|v| v as i8);
            DateTime::new(t.year.try_into().ok()?, mo, d, h, mi, s, 0).ok()
        });
    let camera = [text(Tag::Make), text(Tag::Model)]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    Exif {
        taken,
        camera: (!camera.is_empty()).then_some(camera),
        gps: exif.get_field(Tag::GPSLatitude, In::PRIMARY).is_some(),
    }
}

#[cfg(test)]
mod tests {
    use exif::Field;
    use image::{DynamicImage, ImageFormat};
    use jiff::civil::date;

    use super::*;
    use crate::testing::{Temp, encode, jpeg_with_exif};

    fn dt(y: i16, m: i8, d: i8) -> DateTime {
        date(y, m, d).at(0, 0, 0, 0)
    }

    fn noon(y: i16, m: i8, d: i8) -> DateTime {
        date(y, m, d).at(12, 0, 0, 0)
    }

    #[test]
    fn pixel_bounds_are_inclusive() {
        let r = pixel_range("100..200").unwrap();
        assert!(!r.contains(99) && r.contains(100) && r.contains(200) && !r.contains(201));
        let exact = pixel_range("640").unwrap();
        assert!(exact.contains(640) && !exact.contains(641) && !exact.contains(639));
        assert!(pixel_range("1920..").unwrap().contains(u32::MAX - 1));
        assert!(pixel_range("..10").unwrap().contains(0));
    }

    #[test]
    fn invalid_ranges() {
        for bad in ["", "..", "abc", "5..3", "1..x", "-5", "1...2"] {
            assert!(pixel_range(bad).is_err(), "{bad:?} should fail");
        }
        assert!(pixel_range("5..5").is_ok());
    }

    #[test]
    fn date_bounds_cover_whole_period() {
        let r = time_range("2024-06..2024-08").unwrap();
        assert_eq!(
            r,
            Range {
                lo: Some(dt(2024, 6, 1)),
                hi: Some(dt(2024, 9, 1))
            }
        );
        assert_eq!(
            time_range("2023").unwrap(),
            Range {
                lo: Some(dt(2023, 1, 1)),
                hi: Some(dt(2024, 1, 1))
            }
        );
        let day = time_range("2024-02-29").unwrap();
        assert!(day.contains(date(2024, 2, 29).at(23, 59, 59, 0)));
        assert!(!day.contains(dt(2024, 3, 1)));
        assert_eq!(time_range("2024-12..").unwrap().lo, Some(dt(2024, 12, 1)));
        assert_eq!(time_range("..2024-12").unwrap().hi, Some(dt(2025, 1, 1)));

        let t = |h, mi, s, ns| date(2024, 6, 15).at(h, mi, s, ns);
        for (s, lo, hi) in [
            ("2024-06-15T14", t(14, 0, 0, 0), t(15, 0, 0, 0)),
            ("2024-06-15 14:30", t(14, 30, 0, 0), t(14, 31, 0, 0)),
            ("2024-06-15t14:30:05", t(14, 30, 5, 0), t(14, 30, 6, 0)),
            (
                "2024-06-15T14:30:05.25",
                t(14, 30, 5, 250_000_000),
                t(14, 30, 5, 260_000_000),
            ),
            (
                "2024-06-15T23:59:59.999999999",
                t(23, 59, 59, 999_999_999),
                dt(2024, 6, 16),
            ),
        ] {
            assert_eq!(time_range(s), period(lo, hi), "{s}");
        }
        assert_eq!(
            time_range("2024-06-15 14:30..2024-06-15T15").unwrap(),
            Range {
                lo: Some(t(14, 30, 0, 0)),
                hi: Some(t(16, 0, 0, 0))
            }
        );
    }

    /// Parses `s` at noon in Prague on 2026-03-29, the day the clocks go forward an hour.
    fn at(s: &str) -> Result<Range<DateTime>, String> {
        time_range_at(s, &"2026-03-29T12:00[Europe/Prague]".parse().unwrap())
    }

    fn period(lo: DateTime, hi: DateTime) -> Result<Range<DateTime>, String> {
        Ok(Range {
            lo: Some(lo),
            hi: Some(hi),
        })
    }

    #[test]
    fn offsets_become_local_time() {
        let t = |h, mi| date(2024, 6, 15).at(h, mi, 0, 0);
        assert_eq!(at("2024-06-15T14:30Z"), period(t(16, 30), t(16, 31)));
        assert_eq!(at("2024-06-15t14:30z"), period(t(16, 30), t(16, 31)));
        assert_eq!(at("2024-06-15T14:30+02:00"), period(t(14, 30), t(14, 31)));
        assert_eq!(at("2024-06-15 14:00-05:30"), period(t(21, 30), t(21, 31)));
        assert_eq!(at("2024-06-15T14+02"), period(t(14, 0), t(15, 0)));
        let winter = date(2024, 1, 15);
        assert_eq!(
            at("2024-01-15T14Z"),
            period(winter.at(15, 0, 0, 0), winter.at(16, 0, 0, 0))
        );
    }

    #[test]
    fn calendar_dates_ignore_daylight_saving() {
        let t = |d, h, mi| date(2026, 3, d).at(h, mi, 0, 0);
        assert_eq!(at("2026-03-29T02"), period(t(29, 2, 0), t(29, 3, 0)));
        assert_eq!(at("2026-03-29T02:30"), period(t(29, 2, 30), t(29, 2, 31)));
        assert_eq!(at("2026-03-29"), period(t(29, 0, 0), t(30, 0, 0)));
        assert_eq!(at("today"), period(t(29, 0, 0), t(30, 0, 0)));
        assert_eq!(at("yesterday"), period(t(28, 0, 0), t(29, 0, 0)));
        assert_eq!(at("yesterday..").unwrap().hi, None);
        assert_eq!(at("..yesterday").unwrap().hi, Some(t(29, 0, 0)));
        assert_eq!(at("1d").unwrap().lo, Some(t(28, 12, 0)));
        assert_eq!(at("24h").unwrap().lo, Some(t(28, 11, 0)));
    }

    #[test]
    fn ages_count_back_from_now() {
        let ago = |s| at(s).unwrap().lo.unwrap();
        assert_eq!(ago("15m"), date(2026, 3, 29).at(11, 45, 0, 0));
        assert_eq!(ago("1h30m"), date(2026, 3, 29).at(10, 30, 0, 0));
        assert_eq!(ago("1h 30m ago"), date(2026, 3, 29).at(10, 30, 0, 0));
        assert_eq!(ago("2 days"), noon(2026, 3, 27));
        assert_eq!(ago("7d ago"), noon(2026, 3, 22));
        assert_eq!(ago("2w"), noon(2026, 3, 15));
        assert_eq!(ago("3mo"), noon(2025, 12, 29));
        assert_eq!(ago("1 year"), noon(2025, 3, 29));
        let week = Some(noon(2026, 3, 22));
        assert_eq!(at("7d"), Ok(Range { lo: week, hi: None }));
        assert_eq!(at("7d.."), Ok(Range { lo: week, hi: None }));
        assert_eq!(at("..7d"), Ok(Range { lo: None, hi: week }));
        assert_eq!(at("1y..7d"), period(noon(2025, 3, 29), noon(2026, 3, 22)));
        assert_eq!(at("2026..7d"), period(dt(2026, 1, 1), noon(2026, 3, 22)));
        assert!(time_range("7d").unwrap().contains(Zoned::now().datetime()));
    }

    #[test]
    fn date_errors_teach_the_grammar() {
        let err = |s| at(s).unwrap_err();
        let sign = "an age has no sign: 7d.. for newer, ..7d for older";
        assert_eq!(err("-7d.."), sign);
        assert_eq!(err("..+7d"), sign);
        let case = "units are lowercase: mo for months, m for minutes";
        assert_eq!(err("3M"), case);
        assert_eq!(err("7D.."), case);
        assert!(err("2024Z").contains("is not a date"));
        assert_eq!(err("2025..2024"), "start is after end");
        assert_eq!(err("today..yesterday"), "start is after end");
        assert_eq!(err("7d..2025"), "start is after end");
        assert_eq!(err("1d..1y"), "ages go oldest first: 1y..1d");
        assert_eq!(err(".."), "expected A..B, A.., ..B or A");
        for bad in [
            "",
            "24",
            "7",
            "7x",
            "d",
            "2024-13",
            "2023-02-29",
            "2024-6",
            "20240615",
            "2024-06-15T1430",
            "2024-06-15T14:30[Europe/Prague]",
            "14:30",
            "2024-06-15T24:00",
            "2024-06-15T14:60",
            "2024-06-15  14:30",
            "2024-06-15T14:30:05.",
            "2024-06-15T14:30:05.1234567890",
            "2024-06-15Z",
            "2024-06-15T14+0200",
            "2024-06-15T14+02:60",
            "1h  30m",
            "7dago",
            "1.5h",
            "1h, 30m",
            "p1d",
            "Today",
            "now",
            "tomorrow",
        ] {
            assert_eq!(
                err(bad),
                format!(
                    "'{bad}' is not a date (2024-06-15T14:30), an age (7d, 1h30m), today or \
                     yesterday"
                )
            );
        }
    }

    fn meta(exif: Exif) -> Meta {
        Meta {
            width: 4000,
            height: 3000,
            exif,
        }
    }

    /// A file time in the local time zone, as the file system gives it.
    fn local(t: DateTime) -> SystemTime {
        t.to_zoned(jiff::tz::TimeZone::system())
            .unwrap()
            .timestamp()
            .into()
    }

    fn check(f: &MetaFilter, exif: Exif) -> bool {
        f.check(&meta(exif), local(noon(2020, 1, 1)))
    }

    #[test]
    fn meta_filters() {
        let canon = Exif {
            taken: Some(dt(2024, 7, 4)),
            camera: Some("Canon EOS R6".into()),
            gps: true,
        };
        assert!(!MetaFilter::default().needs_meta());
        assert!(check(&MetaFilter::default(), Exif::default()));

        let f = MetaFilter {
            width: pixel_range("3000..").ok(),
            height: pixel_range("..3000").ok(),
            ..Default::default()
        };
        assert!(f.needs_meta() && check(&f, Exif::default()));
        let narrow = MetaFilter {
            width: pixel_range("..3999").ok(),
            ..Default::default()
        };
        assert!(!check(&narrow, Exif::default()));

        let f = MetaFilter {
            camera: regex::RegexBuilder::new("canon r6|eos")
                .case_insensitive(true)
                .build()
                .ok(),
            ..Default::default()
        };
        assert!(check(&f, canon.clone()));
        assert!(!check(&f, Exif::default()));

        let gps = |want| MetaFilter {
            gps: Some(want),
            ..Default::default()
        };
        assert!(check(&gps(true), canon.clone()));
        assert!(!check(&gps(false), canon));
        assert!(check(&gps(false), Exif::default()));
    }

    #[test]
    fn date_is_taken_else_modified() {
        let summer = MetaFilter {
            date: time_range("2024-06..2024-08").ok(),
            ..Default::default()
        };
        let taken = |t| Exif {
            taken: Some(t),
            ..Default::default()
        };
        let july = local(noon(2024, 7, 10));
        assert!(summer.needs_meta());
        assert!(summer.check(&meta(Exif::default()), july));
        assert!(summer.check(&meta(taken(dt(2024, 8, 31))), local(noon(2020, 1, 1))));
        assert!(!summer.check(&meta(taken(dt(2024, 9, 1))), july));
        let plain = meta(Exif::default());
        assert_eq!(date_of(Some(&plain), july), Some(noon(2024, 7, 10)));
        assert_eq!(date_of(None, july), Some(noon(2024, 7, 10)));
        let late = july + std::time::Duration::from_millis(900);
        assert_eq!(
            date_of(None, late),
            Some(noon(2024, 7, 10)),
            "whole seconds"
        );
    }

    fn ascii(tag: Tag, s: &str) -> Field {
        Field {
            tag,
            ifd_num: In::PRIMARY,
            value: Value::Ascii(vec![s.as_bytes().to_vec()]),
        }
    }

    fn read_bytes(bytes: &[u8], name: &str) -> Result<Meta> {
        read(&Temp::file(name, bytes))
    }

    #[test]
    fn reads_exif() {
        let gps = Field {
            tag: Tag::GPSLatitude,
            ifd_num: In::PRIMARY,
            value: Value::Rational(vec![(50, 1).into(), (5, 1).into(), (0, 1).into()]),
        };
        let orientation = Field {
            tag: Tag::Orientation,
            ifd_num: In::PRIMARY,
            value: Value::Short(vec![6]),
        };
        let bytes = jpeg_with_exif(
            3,
            2,
            &[
                ascii(Tag::Make, "Canon"),
                ascii(Tag::Model, "EOS R6"),
                ascii(Tag::DateTime, "2020:01:01 00:00:00"),
                ascii(Tag::DateTimeOriginal, "2024:07:04 18:30:05"),
                gps,
                orientation,
            ],
        );
        let meta = read_bytes(&bytes, "exif.jpg").unwrap();
        assert_eq!(
            meta,
            Meta {
                width: 2,
                height: 3,
                exif: Exif {
                    taken: Some(date(2024, 7, 4).at(18, 30, 5, 0)),
                    camera: Some("Canon EOS R6".into()),
                    gps: true
                },
            }
        );
    }

    #[test]
    fn exif_fallbacks() {
        let bytes = jpeg_with_exif(
            3,
            2,
            &[
                ascii(Tag::DateTime, "2021:02:03 04:05:06"),
                ascii(Tag::Model, "Pixel 8"),
            ],
        );
        let exif = read_bytes(&bytes, "fallback.jpg").unwrap().exif;
        assert_eq!(
            exif,
            Exif {
                taken: Some(date(2021, 2, 3).at(4, 5, 6, 0)),
                camera: Some("Pixel 8".into()),
                gps: false
            }
        );

        let bytes = jpeg_with_exif(
            3,
            2,
            &[
                ascii(Tag::DateTime, "2025:01:01 00:00:00"),
                ascii(Tag::DateTimeDigitized, "2021:02:03 04:05:06"),
            ],
        );
        assert_eq!(
            read_bytes(&bytes, "digitized.jpg").unwrap().exif.taken,
            Some(date(2021, 2, 3).at(4, 5, 6, 0))
        );

        let bytes = jpeg_with_exif(3, 2, &[ascii(Tag::DateTimeOriginal, "0000:00:00 00:00:00")]);
        assert_eq!(
            read_bytes(&bytes, "zero.jpg").unwrap().exif,
            Exif::default()
        );
    }

    #[test]
    fn missing_or_broken_exif_is_default() {
        let png = encode(&DynamicImage::new_rgb8(5, 7), ImageFormat::Png);
        assert_eq!(
            read_bytes(&png, "plain.png").unwrap(),
            Meta {
                width: 5,
                height: 7,
                exif: Exif::default()
            }
        );

        let mut broken = jpeg_with_exif(3, 2, &[ascii(Tag::Make, "Canon")]);
        broken[12..20].fill(0xAB);
        assert_eq!(
            read_bytes(&broken, "broken.jpg").unwrap().exif,
            Exif::default()
        );

        assert!(read_bytes(b"not an image", "junk.png").is_err());
    }
}
