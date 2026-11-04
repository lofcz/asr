const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 for(const [name,url] of [['spotmate','https://spotmate.online/en1'],['spotsaver','https://spotsaver.net/']]){
  const context=await browser.newContext();const page=await context.newPage();const log=[];
  page.on('response',async response=>{const req=response.request();const entry={url:response.url(),status:response.status(),type:req.resourceType(),method:req.method()};log.push(entry);
   if(['script','xhr','fetch'].includes(req.resourceType()) && new URL(response.url()).hostname===new URL(url).hostname){try{const body=await response.body();fs.writeFileSync(`${__dirname}/${name}-network-${log.length}.txt`,body)}catch{}}
  });
  try{
   await page.goto(url,{waitUntil:'domcontentloaded',timeout:30000});await page.waitForTimeout(5000);
   console.log(name,JSON.stringify(await page.locator('input').evaluateAll(es=>es.map(e=>({type:e.type,placeholder:e.placeholder,id:e.id,name:e.name})))));
   console.log(name,'buttons',JSON.stringify(await page.locator('button').allTextContents()));
   await page.locator('input[type=text]').first().fill('https://open.spotify.com/episode/28OzoDZKpgDbEAGYonrkoo');
   if(name==='spotmate') await page.getByRole('button',{name:/^Start/}).click();
   else await page.getByRole('button',{name:'Download',exact:true}).click();
   await page.waitForTimeout(12000);
   fs.writeFileSync(`${__dirname}/${name}-submitted.txt`,await page.locator('body').innerText());
   console.log(name,'RESULT',(await page.locator('body').innerText()).slice(0,2200));
   await page.screenshot({path:`${__dirname}/${name}.png`,fullPage:false});
  }catch(e){console.log(name,String(e))}
  fs.writeFileSync(`${__dirname}/${name}-network.json`,JSON.stringify(log,null,2));await context.close();
 }
 await browser.close();
})();
