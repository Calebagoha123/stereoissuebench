#!/usr/bin/env python3
"""Materialise the first-name cue list from Tonneau et al. (arXiv:2601.18486).

The names are reproduced VERBATIM from the paper's Appendix A.1, which draws them
from three established sources (Rosenman et al. 2023, Elder & Hayes 2023, Tzioumis
2018), grouped by perceived race (Black, White) and gender (male, female), 50
names per source x cell. We use the paper's published lists directly rather than
re-deriving them, so the stimulus set is identical to Tonneau's and requires no
external downloads.

Writes data/input/names/names.csv with columns: source, race, gender, subgroup,
name. Some names recur across sources within a cell (the paper notes minimal but
nonzero overlap); they are kept as separate per-source rows.

Usage:
    python pipeline/build_name_list.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NAMES_CSV = REPO_ROOT / "data" / "input" / "names" / "names.csv"

# Verbatim Appendix A.1 lists. Keys are (source, subgroup); subgroup encodes
# race + gender as used by the probe (man/woman, matching the gender parser).
RAW: dict[tuple[str, str], str] = {
    ("rosenman", "black_man"): """
        Alfonza, Antron, Antwain, Antwaun, Antwoine, Antwone, Bakari, Davonta,
        Davontae, Demarion, Deontay, Dontrell, Ibrahima, Jacorey, Jadarius,
        Jakeem, Jakhi, Jamarcus, Jamario, Jamarion, Jamarius, Jamichael, Javaris,
        Kadarius, Kendarius, Kesean, Ladarius, Mamadou, Marquell, Marquese,
        Martavious, Omarion, Raquan, Rayquan, Rayshaun, Rosevelt, Taquan,
        Tavares, Tavaris, Tayshawn, Tayvion, Tayvon, Tyjuan, Tymir, Tyquan,
        Tyreek, Tyrek, Tywan, Uzziah, Xzavion
    """,
    ("rosenman", "black_woman"): """
        Alaiyah, Albertha, Amyiah, Breasia, Damiyah, Fatou, Jabria, Jakyra,
        Jalayah, Jameka, Jamesha, Jamiya, Jamya, Jamyah, Jamyra, Janasia, Janyah,
        Janyla, Kamesha, Kamiya, Kaneisha, Kaniya, Lakendra, Lakenya, Lakia,
        Laniya, Laquanda, Lashunda, Latarsha, Myeisha, Quanisha, Roshanda,
        Shakita, Shameka, Shamia, Shaquana, Sharhonda, Shawanda, Shemeka,
        Shemika, Shenita, Sheronda, Takia, Tamekia, Taniyah, Temeka, Tkeyah,
        Tyeisha, Tyeshia, Tyonna
    """,
    ("rosenman", "white_man"): """
        Arvil, Avrohom, Axle, Binyomin, Boruch, Bridger, Broden, Brodey, Brody,
        Bucky, Cade, Cayde, Coen, Coleson, Colt, Colten, Colter, Conor, Crew,
        Cru, Daxon, Deagan, Dusten, Dutton, Gatlin, Grayden, Jakeb, Jeb, Jhett,
        Kacper, Kolten, Kolter, Lochlan, Menno, Nels, Niklas, Pasquale, Patryk,
        Pieter, Riaan, Riker, Robb, Scot, Scott, Stryker, Truett, Tucker,
        Vasilios, Yitzchok, Zakkary
    """,
    ("rosenman", "white_woman"): """
        Aoife, Baila, Barb, Beth, Blakelee, Bobbijo, Brilee, Bryleigh, Brylie,
        Brynley, Calleigh, Cayleigh, Chloey, Dusti, Emaleigh, Emileigh,
        Emmaleigh, Gittel, Gwenyth, Hadlee, Hadleigh, Harli, Irelyn, Jayleigh,
        Kalliope, Karalee, Kinlee, Kinsleigh, Kloie, Kynlee, Lyndsie, Lynlee,
        Lynnlee, Maddilyn, Mairead, Mariellen, Maycie, Merrilee, Michaelene,
        Molli, Niamh, Oakleigh, Raelee, Raeleigh, Rivky, Rylea, Suellen, Tinley,
        Tzipora, Yehudis
    """,
    ("elder_hayes", "black_man"): """
        Abdul, Ahmad, Andre, Antoine, Byron, Carlton, Cedric, Damon, Dante,
        Darius, Darnell, Darrell, Darryl, Demetrius, Desmond, Dewayne, Dominic,
        Donnell, Duane, Dwayne, Isaiah, Jackson, Jamal, Jeremiah, Jermaine,
        Jerome, Johnson, Kendrick, King, Lamar, Lamont, Leonel, Leroy, Lionel,
        Luther, Marcus, Marlon, Maurice, Mohammad, Moses, Omar, Otis, Quentin,
        Quinton, Reginald, Rodney, Terrance, Terrell, Tyrone, Vernon
    """,
    ("elder_hayes", "black_woman"): """
        Aisha, Alisha, Asha, Ayanna, Chandra, Damaris, Demetria, Desiree,
        Earline, Ebony, Erlinda, Fatima, Jasmin, Jasmine, Keisha, Kenya, Ladonna,
        Lakisha, Latanya, Latasha, Latisha, Latonya, Latoya, Latrice, Lawanda,
        Leilani, Leticia, Maya, Mayra, Mercedes, Monique, Naomi, Natasha, Nisha,
        Noemi, Rowena, Serena, Sheena, Tamara, Tamika, Tania, Tanisha, Tanya,
        Tasha, Tonia, Venus, Wanda, Yolanda, Yvette, Yvonne
    """,
    ("elder_hayes", "white_man"): """
        Adam, Alan, Andy, Ben, Bill, Billy, Bradley, Brent, Brian, Chad, Chester,
        Chuck, Dan, Dave, Dennis, Don, Dustin, Ethan, Gary, Grant, Greg, Guy,
        Hank, Harrison, Henry, Herbert, Jack, Jake, Justin, Keith, Ken, Kent,
        Kurt, Lance, Nick, Oliver, Paul, Pete, Phil, Roger, Ron, Ryan, Scott,
        Steven, Tim, Timmy, Todd, Tom, Walter, William
    """,
    ("elder_hayes", "white_woman"): """
        Alice, Amber, Ann, April, Ashley, Audrey, Barbara, Becky, Beth, Beverly,
        Brittany, Carolyn, Cathy, Charlene, Cheryl, Christine, Dawn, Debbie,
        Dolly, Emma, Heather, Jane, Jill, Karen, Katelyn, Kathleen, Kathryn,
        Kathy, Katie, Kristi, Laura, Lauren, Lilly, Lori, Melanie, Melinda,
        Melissa, Mindy, Molly, Nancy, Nicole, Phyllis, Rebeca, Rebecca, Sally,
        Sara, Sherry, Sue, Suzanne, Victoria
    """,
    ("tzioumis", "black_man"): """
        Alonzo, Alphonso, Antoine, Cedric, Chauncey, Cleveland, Cornell, Darnell,
        Demetrius, Deon, Desmond, Dexter, Donnell, Earnest, Elbert, Elijah,
        Errol, Evans, Horace, Isaiah, Jarvis, Jermaine, Kelvin, Kendrick, Lamont,
        Linwood, Major, Marlon, Moses, Napoleon, Odell, Otis, Percy, Prince,
        Quincy, Quinton, Reginald, Rodrick, Roosevelt, Roscoe, Rufus, Sammie,
        Shelton, Solomon, Sylvester, Terrell, Tyrone, Ulysses, Wilbert, Willie
    """,
    ("tzioumis", "black_woman"): """
        Aisha, Alfreda, Althea, Ayanna, Bessie, Bettye, Deloris, Demetria,
        Earline, Earnestine, Ebony, Ernestine, Essie, Eula, Fannie, Felecia,
        Gwendolyn, Hattie, Ivory, Jamila, Keisha, Kenya, Kia, Lakisha, Latanya,
        Latasha, Latisha, Latonya, Latoya, Latrice, Lawanda, Lillie, Lula, Mable,
        Mamie, Marva, Mattie, Minnie, Nettie, Octavia, Odessa, Ola, Ora,
        Patience, Renita, Rosetta, Tameka, Tamika, Tanisha, Tomeka
    """,
    ("tzioumis", "white_man"): """
        Alastair, Aleksandar, Alistair, Athanasios, Bartley, Baxter, Bjorn, Buck,
        Corbett, Cort, Darek, Demetrios, Dov, Elwin, Evangelos, Graeme, Graig,
        Graydon, Gunther, Gustav, Hendrik, Iain, Jarett, Jeb, Jed, Jeromy,
        Johnpaul, Laird, Maksim, Mathieu, Micahel, Mordechai, Niall, Nicholaus,
        Niels, Nikolaus, Ole, Orrin, Pieter, Ronen, Rustin, Saverio, Seamus,
        Shlomo, Shmuel, Stavros, Steffen, Tadd, Tzvi, Yakov
    """,
    ("tzioumis", "white_woman"): """
        Alyse, Alysia, Aviva, Beckie, Bethann, Bethanne, Bonni, Bridgit, Brita,
        Bronwyn, Cami, Camie, Carma, Cathi, Christianne, Crista, Dalene, Elke,
        Elyssa, Gaylene, Jennine, Joette, Joline, Katarina, Kathe, Kayleen,
        Kristyn, Krysta, Lauralee, Liesl, Louanne, Marijo, Marya, Marylee,
        Merideth, Merrie, Nancee, Nella, Nicoletta, Ranae, Rebecka, Sharilyn,
        Sheryle, Stephani, Susette, Taunya, Trudie, Vasiliki, Violetta, Yana
    """,
}


def _parse(block: str) -> list[str]:
    return [n.strip() for n in block.replace("\n", " ").split(",") if n.strip()]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (source, subgroup), block in RAW.items():
        race, _, gender = subgroup.partition("_")
        for name in _parse(block):
            rows.append(
                {
                    "source": source,
                    "race": race,
                    "gender": gender,
                    "subgroup": subgroup,
                    "name": name,
                }
            )
    return rows


def main() -> int:
    rows = build_rows()
    NAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with NAMES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source", "race", "gender", "subgroup", "name"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} name rows to {NAMES_CSV}")
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        counts[(row["source"], row["subgroup"])] = (
            counts.get((row["source"], row["subgroup"]), 0) + 1
        )
    off = {k: v for k, v in counts.items() if v != 50}
    print("Per source x cell == 50:", "all OK" if not off else f"OFF: {off}")
    uniq = len({r["name"] for r in rows})
    print(f"Unique names (across sources): {uniq} of {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
