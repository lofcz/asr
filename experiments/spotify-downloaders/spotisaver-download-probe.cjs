const {chromium}=require('playwright');const fs=require('fs');
const episode='https://open.spotify.com/episode/28OzoDZKpgDbEAGYonrkoo';
(async()=>{const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
for(const [name,url,input,button] of [['spotisaver','https://spotisaver.net/en1','#playlistUrl','Load']]){
 const context=await browser.newContext();const page=await context.newPage();const log=[];const pending=[];
 page.on('response',response=>{const req=response.request();if(!['xhr','fetch'].includes(req.resourceType())||!response.url().startsWith(new URL(url).origin))return;
 const record={url:response.url(),method:req.method(),status:response.status()};log.push(record);
 pending.push((async()=>{try{record.body=(await response.text()).slice(0,12000)}catch{}})());
 });
 try{
  await page.goto(url,{waitUntil:'domcontentloaded',timeout:25000});await page.waitForTimeout(2500);
  await page.locator(input).first().fill(episode);await page.getByRole('button',{name:button,exact:true}).click({timeout:8000});
  await page.waitForTimeout(8000);
  if(name==='spotisaver') {await page.locator('.btn-download').first().click({timeout:25000});await page.waitForTimeout(18000);}
  const text=await page.locator('body').innerText();fs.writeFileSync(`${__dirname}/${name}-action.txt`,text);
  console.log(name,text.slice(0,1800));
 }catch(e){console.log(name,String(e));fs.writeFileSync(`${__dirname}/${name}-error.txt`,await page.locator('body').innerText());}
 await Promise.allSettled(pending);fs.writeFileSync(`${__dirname}/${name}-action-network.json`,JSON.stringify(log,null,2));await context.close();
}await browser.close();})();
