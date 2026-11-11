use std::{path::Path, time::SystemTime};

use anyhow::Result;
use exif::{In, Tag, Value};
use jiff::{
    Span, Zoned,
    civil::{Date, DateTime},
};
use regex::Regex;

use crate::decode;

#[derive(Debug, Clone, PartialEq)]
pub struct Meta {
    pub width: u32,
    pub height: u32,
    pub exif: Exif,
}

impl Meta {
    /// When the photo was taken, else when the file last changed. Phone galleries do the same, so
    /// screenshots and downloads get a date too.
    pub fn date(&self, modified: SystemTime) -> Option<DateTime> {
        (self.exif.taken).or_else(|| Some(Zoned::try_from(modified).ok()?.datetime()))
    }
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

    /// `period` maps one bound to the `[start, end)` it covers.
    fn parse(s: &str, period: impl Fn(&str) -> Result<(T, T), String>) -> Result<Self, String> {
        let bound = |b: &str| (!b.is_empty()).then(|| period(b)).transpose();
        let (lo, hi) = match s.split_once("..") {
            Some((a, b)) => (bound(a)?.map(|p| p.0), bound(b)?.map(|p| p.1)),
            None => period(s).map(|(a, b)| (Some(a), Some(b)))?,
        };
        match (lo, hi) {
            (None, None) => Err("expected A..B, A.., ..B or A".into()),
            (Some(lo), Some(hi)) if lo >= hi => Err("start is after end".into()),
            _ => Ok(Range { lo, hi }),
        }
    }
}

pub fn pixel_range(s: &str) -> Result<Range<u32>, String> {
    Range::parse(s, |p| {
        let n: u32 = p
            .parse()
            .map_err(|_| format!("'{p}' is not a pixel count"))?;
        Ok((n, n.saturating_add(1)))
    })
}

pub fn time_range(s: &str) -> Result<Range<DateTime>, String> {
    if !s.contains("..") && s.ends_with(|c: char| c.is_ascii_alphabetic()) {
        return Err(format!(
            "an age needs a side: {s}.. for newer, ..{s} for older"
        ));
    }
    let now = Zoned::now();
    Range::parse(s, |p| time_period(p, &now))
}

fn time_period(s: &str, now: &Zoned) -> Result<(DateTime, DateTime), String> {
    let invalid = || format!("'{s}' is not a date (YYYY[-MM[-DD]]) or an age (7d, 3mo, 1y)");
    if let Some(i) = s.find(|c: char| c.is_ascii_alphabetic()) {
        let n = i64::from(s[..i].parse::<u32>().map_err(|_| invalid())?);
        let span = match &s[i..] {
            "h" => Span::new().try_hours(n),
            "d" => Span::new().try_days(n),
            "w" => Span::new().try_weeks(n),
            "mo" => Span::new().try_months(n),
            "y" => Span::new().try_years(n),
            _ => return Err(invalid()),
        };
        let t = span
            .and_then(|span| now.checked_sub(span))
            .map_err(|e| e.to_string())?
            .datetime();
        return Ok((t, t));
    }
    let (full, period) = match s.len() {
        4 => (format!("{s}-01-01"), Span::new().years(1)),
        7 => (format!("{s}-01"), Span::new().months(1)),
        10 => (s.to_string(), Span::new().days(1)),
        _ => return Err(invalid()),
    };
    let start: Date = full.parse().map_err(|_| invalid())?;
    let end = start.checked_add(period).map_err(|e| e.to_string())?;
    Ok((start.into(), end.into()))
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
            && within(&self.date, || meta.date(modified))
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
    let taken = [Tag::DateTimeOriginal, Tag::DateTime]
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
        for bad in [
            "24",
            "2024-13",
            "2023-02-29",
            "2024-6",
            "20240601",
            "7",
            "7x",
            "d",
            "2025..2024",
        ] {
            assert!(time_range(bad).is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn ages_count_back_from_now() {
        let now: Zoned = "2026-03-15T12:00[UTC]".parse().unwrap();
        let ago = |s| time_period(s, &now).unwrap().0;
        assert_eq!(ago("36h"), date(2026, 3, 14).at(0, 0, 0, 0));
        assert_eq!(ago("7d"), noon(2026, 3, 8));
        assert_eq!(ago("2w"), noon(2026, 3, 1));
        assert_eq!(ago("1mo"), noon(2026, 2, 15));
        assert_eq!(ago("1y"), noon(2025, 3, 15));
        assert!(time_range("7d").is_err() && time_range("-7d..").is_err());
        let last_week = time_range("7d..").unwrap();
        assert!(last_week.contains(Zoned::now().datetime()));
        assert!(time_range("1y..1d").is_ok() && time_range("1d..1y").is_err());
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
        assert_eq!(meta(Exif::default()).date(july), Some(noon(2024, 7, 10)));
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
