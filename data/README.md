
## Data Preparation

Building blocks are **not distributed** with this repository due to licensing restrictions. 
1. Please follow the steps below to prepare your building block data:

- **Enamine building blocks (optional):**  
      If you want to use Enamine building blocks, you need to download them directly from  
        [https://enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog),  
        which requires approval.

- **Alternatively, ZINC building blocks:**  

 **Download building blocks from ZINC12**  
   Visit the following URL and download an appropriate building block catalog:  
   [ZINC12 Building Blocks](https://zinc12.docking.org/browse/catalogs/building-blocks)

2. **Rename the downloaded file**  
   Rename your downloaded file to:  


3. **Move the file to the data directory**  
Place the `building_blocks.csv` file inside the `data/` folder of this repository.

4. **Ensure correct CSV format**  
The CSV file must have the following **column names**:  



5. **Run the data preparation script**  
Execute the following command to process the building blocks and prepare them for FragDockRL:  
```bash
python data/preparation.py


