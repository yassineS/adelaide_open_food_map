# Open Food Map

This project provides tools to collect, analyse, and visualise restaurant data. It is designed to be modular and extensible, allowing you to easily add support for new cities.

## Features

* **Data Collection**: Scrape restaurant data from Google Maps API using a grid-based search.
* **Analysis**: Identify "underrated" restaurants using a machine learning model that compares ratings to review counts and other factors.
* **Visualisation**: Generate interactive HTML maps to explore the culinary landscape.

## Project Status & Limitations

This project is a work in progress.

* Coverage reflects what the Google Maps API surfaces in generic searches; it is not a census of all restaurants.
* Rating models capture platform-typical outcomes, not intrinsic food quality.
* Restaurant age, pricing, review text, and reviewer composition are not yet fully incorporated.
* Results should be interpreted as exploratory and comparative, not definitive rankings.

Contributions, critiques, and extensions are very welcome.

## Project Structure

* `src/`: Source code.
  * `data_collection/`: Scripts for gathering data.
  * `analysis/`: Scripts for processing and analyzing data.
  * `visualisation/`: Scripts for creating maps and plots.
* `data/`: Data storage.
  * `raw/`: Raw collected data.
  * `processed/`: Analysed and cleaned data.
  * `sample/`: Small sample datasets for testing.
* `tests/`: Unit tests.

## Setup

### Google API requirements

The data collection scripts require the **Google Places API (Legacy)**.

**Important:** You must enable the **legacy "Places API"** (NOT "Places API (New)") in your Google Cloud Console.

**To enable the Places API (Legacy):**

1. Go to [Google Cloud Console - API Library](https://console.cloud.google.com/apis/library)
2. Search for "**Places API**" (make sure it's NOT "Places API (New)")
3. Click on "**Places API**" and press "**ENABLE**"
4. Ensure your API key has access:
   - Go to [Google Cloud Console - Credentials](https://console.cloud.google.com/apis/credentials)
   - Click on your API key
   - Under "API restrictions", ensure "**Places API**" is enabled (or set to "Don't restrict key")

**Note:** Google is deprecating the legacy Places API. If you cannot enable it in your project, you may need to migrate to "Places API (New)" or use a project that has the legacy API enabled.

Enabling only the **Maps JavaScript API** or **Maps Embed API** is not sufficient and will result in data collection failing with a `REQUEST_DENIED` error.

If the Places API is not enabled, the collector will fail immediately with clear error messages.

1. **Clone the repository**:

    ```bash
    git clone https://github.com/laurencleek/open_food_map.git
    cd open_food_map
    ```

2. **Install dependencies using pixi**:

    ```bash
    pixi install
    ```

    Alternatively, if you don't have pixi installed, you can install it from [pixi.sh](https://pixi.sh) or use pip:

    ```bash
    pip install -r requirements.txt
    ```

3. **Environment Variables**:
    Create a `.env` file in the root directory and add your Google Maps API key:

    ```txt
    GOOGLE_MAPS_API_KEY=your_api_key_here
    ```

## Usage

### 1. Data Collection

To collect data for Adelaide, SA (default), run:

```bash
pixi run collect
```

Or using python directly:

```bash
python src/data_collection/collect_data.py --city-name adelaide --lat-min -34.980223 --lat-max -34.840710 --lon-min 138.483725 --lon-max 138.642882
```

To collect data for a different city, specify the coordinates:

```bash
python src/data_collection/collect_data.py --city-name london --lat-min 51.28 --lat-max 51.69 --lon-min -0.51 --lon-max 0.33
```

Arguments:

* `--city-name`: Name of the city (used for file naming).
* `--lat-min`, `--lat-max`, `--lon-min`, `--lon-max`: Bounding box coordinates.
* `--grid-step`: Grid step size in km (default: 1.5).
* `--radius`: Search radius in meters (default: 1500).

### 2. Analysis

To analyse the collected data and find underrated restaurants:

```bash
pixi run analyse
```

Or using python directly:

```bash
python src/analysis/analyse_underrated.py --city-name adelaide
```

This will read from `data/raw/adelaide_restaurant_details.csv` (or `data/sample/adelaide_restaurant_details.csv` if not found) and write:

-> `data/processed/adelaide_hype_adjusted_ratings.csv`

### 3. Visualisation

To generate the interactive dashboard with cuisine filtering and borough analysis:

```bash
pixi run visualise
```

Or using python directly:

```bash
python src/visualisation/interactive_cuisine_map.py --city-name adelaide
```

This requires the analysis step to be completed first (it reads `data/processed/{city}_hype_adjusted_ratings.csv`). It will generate:

-> `output/adelaide_restaurants_interactive.html`

## How to Contribute & Create New Maps

We'd love to see maps for more cities! Here is how you (or others) can contribute:

### 1. Pick a City

Decide on a city you want to map. You will need its bounding box coordinates (Latitude Min/Max, Longitude Min/Max). You can find these easily on websites like [bboxfinder.com](http://bboxfinder.com).

### 2. Run the Collector

Run the collection script with your city's coordinates. For example, for **Paris**:

```bash
python src/data_collection/collect_data.py --city-name paris --lat-min 48.81 --lat-max 48.90 --lon-min 2.22 --lon-max 2.46
```

This will generate `data/raw/paris_restaurant_details.csv`.

### 3. Share Your Data

If you want to combine data with others:
1.**Locate your CSV**: Find the `[city]_restaurant_details.csv` file in `data/raw`.
2.**Merge**: Since the file structure is identical for every city, you can simply combine CSV files (e.g., using Pandas `pd.concat` or Excel) to create a mega-dataset.
3.**Visualise**: Run the visualisation script on your combined file to see a multi-city map!

### 4. Submit a Pull Request

If you've improved the code or want to share a sample dataset for a new city, please submit a Pull Request on GitHub.

## License

This project uses a dual license structure:

* **Original code by Lauren Leek**: [MIT License](LICENSE) (allows commercial use)
* **Modifications by Yassine Souilmi**: [CC BY-NC-SA 4.0](LICENSE) (non-commercial use only)

See the [LICENSE](LICENSE) file for full details. When using or modifying this code, you must:

* Attribute both authors
* Respect the MIT License terms for Lauren's original code
* Respect the CC BY-NC-SA 4.0 terms for Yassine's modifications (non-commercial use only)

## Contact

Created by **Lauren leek**.

If you have any questions or suggestions, please feel free to contact me at [laurencaroline.leek@eui.eu](mailto:laurencaroline.leek@eui.eu).

Modified by **Yassine Souilmi**.

For enquiries email: [hello@ysouilmi.com](mailto:hello@ysouilmi.com)

## Support

This project was written by Lauren Leek, please support here, if you enjoyed this project or found it helpful:

* [☕ Buy her a coffee](https://buymeacoffee.com/laurenleek)
* [📝 Subscribe to her Substack](https://laurenleek.substack.com)
